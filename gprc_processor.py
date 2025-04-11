#!/usr/bin/env python3

import argparse
import asyncio
import json
import math
import os
import time
import subprocess
from typing import Optional, AsyncGenerator, Dict, Tuple

import grpc
import grpc.aio
from transformers import AutoTokenizer
from jetstream.core.proto import jetstream_pb2, jetstream_pb2_grpc
from tqdm.asyncio import tqdm_asyncio
import logging

# --- Constants ---
SHARD_TARGET_PERCENTAGE = 0.90
SHARD_RESULT_SEPARATOR = "\n"
LOCAL_TMP_PATH = "/tmp/input_download.jsonl"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

# --- Template Utilities ---
def load_template(template_file: str) -> str:
    with open(template_file, "r", encoding="utf-8") as f:
        return f.read()

def apply_template(template: str, text: str) -> str:
    return template.format(text=text.strip())

def apply_chat_template(tokenizer, user_text: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return f"User: {user_text}\nAssistant:"
    return tokenizer.apply_chat_template([
        {"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)

# --- Sharding ---
def split_text_into_shards(text: str, num_shards: int) -> list[str]:
    total_len, shards = len(text), []
    shard_len, start = total_len // num_shards, 0
    for i in range(num_shards):
        end = total_len if i == num_shards - 1 else start + shard_len
        while end < total_len and 0x80 <= ord(text[end]) <= 0xBF: end += 1
        shards.append(text[start:end]); start = end
        if start >= total_len and i < num_shards - 1:
            shards.extend([""] * (num_shards - 1 - i)); break
    return shards

# --- GCS Utilities ---
def gs_exists(gs_path: str) -> bool:
    try:
        subprocess.run(["gsutil", "ls", gs_path], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def gs_upload(local_path: str, remote_dir: str):
    try:
        base = os.path.basename(local_path)
        remote_path = os.path.join(remote_dir.rstrip("/"), base)
        subprocess.run(["gsutil", "cp", local_path, remote_path], check=True)
        log.info(f"Uploaded to {remote_path}")
    except Exception as e:
        log.error(f"Failed to upload {local_path} to {remote_dir}: {e}")

# --- Processing Core ---
async def _process_single_chunk(text_chunk, stub, tokenizer, max_output_tokens, max_input_tokens, timeout, debug, line_number, shard_info="") -> Dict:
    result = {"result": None, "error": None, "prompt": None}
    try:
        prompt = apply_chat_template(tokenizer, text_chunk)
        token_count = len(tokenizer.encode(prompt))
        if token_count > max_input_tokens:
            return {"error": f"[ERROR: Chunk exceeded input limit ({token_count})]"}
        request = jetstream_pb2.DecodeRequest(
            text_content=jetstream_pb2.DecodeRequest.TextContent(text=prompt),
            max_tokens=max_output_tokens)
        if debug: result["prompt"] = prompt; start = time.monotonic()
        responses = []
        async for r in stub.Decode(request, timeout=timeout):
            if r and r.stream_content.samples:
                responses.append(r.stream_content.samples[0].text)
        text = "".join(responses).strip()
        if text.endswith("<end_of_turn>"): text = text[:-len("<end_of_turn>")].rstrip()
        result["result"] = text
        if debug: result["processing_time_seconds"] = round(time.monotonic() - start, 3)
    except grpc.aio.AioRpcError as e:
        result["error"] = f"[gRPC ERROR: {e.code()} - {e.details()}]"
    except Exception as e:
        result["error"] = f"[ERROR: {str(e)}]"
    return result

async def process_line_async(stub, tokenizer, line_number, line, max_output_tokens, max_input_tokens, template, debug, semaphore, timeout, task) -> Optional[dict]:
    async with semaphore:
        result = line.copy()
        try:
            text = line.get("text")
            if not text: return {**result, "error": "[ERROR: Missing 'text' field]"}
            templated = apply_template(template, text)
            input_token_count = len(tokenizer.encode(templated))
            if input_token_count <= max_input_tokens:
                chunk = await _process_single_chunk(templated, stub, tokenizer, max_output_tokens, max_input_tokens, timeout, debug, line_number)
                result.update({k: v for k, v in chunk.items() if v is not None})
                if "id" in result: result["id"] += f"_{task}"
                return result

            safe_limit = int(max_input_tokens * SHARD_TARGET_PERCENTAGE)
            if safe_limit <= 0:
                return {**result, "error": "[ERROR: Invalid shard limit]"}

            num_shards = math.ceil(input_token_count / safe_limit)
            shards = split_text_into_shards(templated, num_shards)
            shard_results, total_time, prompts = [], 0, []
            for i, shard in enumerate(shards):
                if not shard: continue
                r = await _process_single_chunk(shard, stub, tokenizer, max_output_tokens, max_input_tokens, timeout, debug, line_number, f"[Shard {i+1}/{num_shards}]")
                if r.get("error"): return {**result, "error": r["error"]}
                shard_results.append(r.get("result", ""))
                if debug: total_time += r.get("processing_time_seconds", 0); prompts.append(r.get("prompt"))
            result["result"] = SHARD_RESULT_SEPARATOR.join(shard_results)
            result["shards_processed"] = num_shards
            if debug: result["processing_time_seconds"] = round(total_time, 3); result["prompt"] = prompts
            if "id" in result: result["id"] += f"_{task}"
        except Exception as e:
            result["error"] = f"[ERROR: {str(e)}]"
        return result

# --- File Utilities ---
async def read_lines_async(filepath: str) -> AsyncGenerator[Tuple[int, Dict], None]:
    with open(filepath, "r", encoding="utf-8") as infile:
        for line_num, line_str in enumerate(infile, 1):
            try: yield line_num, json.loads(line_str)
            except json.JSONDecodeError: log.error(f"Skipping bad JSON at line {line_num}")
            await asyncio.sleep(0)

# --- Main Runner ---
async def run(args):
    if args.input_file.startswith("gs://") and not gs_exists(args.input_file):
        log.critical(f"Input bucket file does not exist: {args.input_file}"); exit(1)

    input_path = args.input_file
    if args.input_file.startswith("gs://"):
        subprocess.run(["gsutil", "cp", args.input_file, LOCAL_TMP_PATH], check=True)
        input_path = LOCAL_TMP_PATH

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    template = load_template(args.template_file)
    address = f"{args.host}:{args.port}"
    semaphore = asyncio.Semaphore(args.concurrent_requests)

    with open(input_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    out = open(args.output_file, "w", encoding="utf-8")

    async with grpc.aio.insecure_channel(address) as channel:
        stub = jetstream_pb2_grpc.OrchestratorStub(channel)
        tasks = [
            asyncio.create_task(process_line_async(
                stub, tokenizer, ln, data, args.max_output_tokens, args.max_input_tokens,
                template, args.debug, semaphore, args.timeout, args.task))
            async for ln, data in read_lines_async(input_path)
        ]

        ok, err, shard, timeout, counter = 0, 0, 0, 0, 0
        for coro in tqdm_asyncio.as_completed(tasks, total=total, desc="Processing"):
            try:
                result = await coro
                counter += 1
                if "error" in result:
                    err += 1
                    if "TIMEOUT" in result["error"]: timeout += 1
                else:
                    ok += 1
                    if "shards_processed" in result: shard += 1
                json.dump(result, out, ensure_ascii=False); out.write("\n")
                if args.output_bucket_dir and counter % args.save_bucket_frequency == 0:
                    out.flush(); gs_upload(args.output_file, args.output_bucket_dir)
            except Exception as e:
                log.error(f"Unhandled error during task completion: {e}")
                err += 1
        out.close()
        if args.output_bucket_dir:
            gs_upload(args.output_file, args.output_bucket_dir)

    log.info(f"Done. OK: {ok}, Errors: {err}, Sharded: {shard}, Timeouts: {timeout}")

# --- CLI Entrypoint ---
def main():
    p = argparse.ArgumentParser("Async JetStream gRPC client")
    p.add_argument("--input_file")
    p.add_argument("--output_file")
    p.add_argument("--template_file")
    p.add_argument("--task", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=9000, type=int)
    p.add_argument("--tokenizer", default="google/gemma-2-9b-it")
    p.add_argument("--max_output_tokens", default=4096, type=int)
    p.add_argument("--max_input_tokens", default=4096, type=int)
    p.add_argument("--concurrent_requests", default=20, type=int)
    p.add_argument("--timeout", default=180.0, type=float)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--output_bucket_dir", default=None)
    p.add_argument("--save_bucket_frequency", type=int, default=50000)
    args = p.parse_args()

    if args.debug: log.setLevel(logging.DEBUG)
    try: asyncio.run(run(args))
    except KeyboardInterrupt: log.info("Interrupted")
    except Exception as e: log.critical(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()

