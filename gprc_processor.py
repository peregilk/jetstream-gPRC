#!/usr/bin/env python3

import argparse
import json
import grpc
from transformers import AutoTokenizer
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from tqdm import tqdm

def format_llama3_prompt(tokenizer, user_prompt: str) -> str:
    """Generate a LLaMA 3-style prompt string using the tokenizer's official chat template."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful, respectful, and honest assistant."},
            {"role": "user", "content": user_prompt.strip()},
        ],
        tokenize=False,
        add_generation_prompt=True
    )

def process_line(stub, tokenizer, line, max_tokens=2000):
    try:
        user_prompt = line["text"]
        prompt = format_llama3_prompt(tokenizer, user_prompt)
        request = jetstream_pb2.DecodeRequest(
            text_content=jetstream_pb2.DecodeRequest.TextContent(text=prompt),
            max_tokens=max_tokens,
        )
        response = stub.Decode(request)
        
        # Collect all response chunks
        full_response = []
        for resp in response:
            chunk = resp.stream_content.samples[0].text
            if chunk:  # Filter empty chunks
                full_response.append(chunk)
        
        return prompt, "".join(full_response).strip()
    except Exception as e:
        return None, f"[ERROR: {str(e)}]"    

def old_process_line(stub, tokenizer, line, max_tokens=2000):
    try:
        user_prompt = line["text"]
        prompt = format_llama3_prompt(tokenizer, user_prompt)
        request = jetstream_pb2.DecodeRequest(
            text_content=jetstream_pb2.DecodeRequest.TextContent(text=prompt),
            max_tokens=max_tokens,
        )
        response = stub.Decode(request)
        output = []
        for resp in response:
            output.extend(resp.stream_content.samples[0].text)
        result = "".join(output).strip()
        return prompt, result
    except Exception as e:
        return None, f"[ERROR: {str(e)}]"

def main():
    parser = argparse.ArgumentParser(description="Process a JSONL corpus using JetStream gRPC with LLaMA 3")
    parser.add_argument("--input_file", default="process.jsonl", help="Input .jsonl file path")
    parser.add_argument("--output_file", default="output.jsonl", help="Output .jsonl file path")
    parser.add_argument("--host", default="localhost", help="gRPC server host")
    parser.add_argument("--port", default=9000, type=int, help="gRPC server port")
    parser.add_argument("--max_tokens", default=2000, type=int, help="Max tokens for completion")
    parser.add_argument("--limit", default=None, type=int, help="Optional limit on number of examples")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tokenizer to use for prompt formatting")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    address = f"{args.host}:{args.port}"
    channel = grpc.insecure_channel(address)
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)

    with open(args.input_file, "r", encoding="utf-8") as infile, \
         open(args.output_file, "w", encoding="utf-8") as outfile:
        for idx, line in enumerate(tqdm(infile, desc="Processing")):
            if args.limit and idx >= args.limit:
                break
            try:
                item = json.loads(line)
                prompt, result = process_line(stub, tokenizer, item, max_tokens=args.max_tokens)
                item["prompt"] = prompt
                item["result"] = result
                json.dump(item, outfile, ensure_ascii=False)
                outfile.write("\n")
            except Exception as e:
                tqdm.write(f"Failed to process line {idx}: {e}")

if __name__ == "__main__":
    main()
