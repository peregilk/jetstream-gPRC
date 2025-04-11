#!/usr/bin/env python3

import argparse
import asyncio
import json
import math # Added for ceiling function
import os
import time
from typing import Optional, AsyncGenerator, Dict, Tuple

import grpc # Keep grpc import for status codes
import grpc.aio # Import the async gRPC module
from transformers import AutoTokenizer
from jetstream.core.proto import jetstream_pb2, jetstream_pb2_grpc
from tqdm.asyncio import tqdm_asyncio
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Percentage of max_input_tokens to target for shard size to leave buffer for chat template overhead
SHARD_TARGET_PERCENTAGE = 0.90
# Separator used when joining results from multiple shards
SHARD_RESULT_SEPARATOR = "\n"


def load_template(template_file: str) -> str:
    """Loads text content from a file."""
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    with open(template_file, "r", encoding="utf-8") as f:
        return f.read()


def apply_template(template: str, text: str) -> str:
    """Applies the loaded template to the input text."""
    return template.format(text=text.strip())


def apply_chat_template(tokenizer, user_text: str) -> str:
    """Applies the tokenizer's chat template to the user text."""
    if tokenizer.chat_template is None:
         logging.warning("Tokenizer does not have a chat_template. Using a basic user prompt format.")
         return f"User: {user_text}\nAssistant:"
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        logging.error(f"Error applying chat template to text: {user_text[:100]}... Error: {e}")
        raise # Re-raise to be caught where called


def split_text_into_shards(text: str, num_shards: int) -> list[str]:
    """Splits a text string into N approximately equal shards based on character length."""
    if num_shards <= 0:
        return []
    if num_shards == 1:
        return [text]

    total_len = len(text)
    shard_len = total_len // num_shards
    shards = []
    start = 0
    for i in range(num_shards):
        # For the last shard, take the remaining text
        end = start + shard_len if i < num_shards - 1 else total_len
        # Adjust end point to avoid splitting in the middle of multi-byte chars (simple approach)
        # This is a basic heuristic, more robust handling might be needed for complex scripts
        while end < total_len and 0x80 <= ord(text[end]) <= 0xBF:
             end += 1
        shards.append(text[start:end])
        start = end
        # Handle case where text is shorter than num_shards
        if start >= total_len and i < num_shards -1:
             # Add empty strings for remaining shards if needed
             shards.extend([""] * (num_shards - 1 - i))
             break
    return shards


async def _process_single_chunk(
    text_chunk: str,
    stub: jetstream_pb2_grpc.OrchestratorStub,
    tokenizer, # Pass tokenizer for chat template
    max_output_tokens: int,
    max_input_tokens: int, # Needed for final safety check
    timeout: float,
    debug: bool,
    line_number: int,
    shard_info: str = "" # e.g., "[Shard 1/3]"
) -> Dict:
    """Processes a single text chunk (either a full line or a shard)."""
    result_dict = {"result": None, "error": None, "prompt": None}
    start_time = time.monotonic() if debug else None

    try:
        # 1. Apply chat template to this specific chunk
        try:
            prompt = apply_chat_template(tokenizer, text_chunk)
        except Exception as e:
             logging.error(f"Line {line_number}{shard_info}: Failed during chat template application for chunk. Error: {e}")
             result_dict["error"] = f"[ERROR: Chat template failed for chunk: {str(e)}]"
             return result_dict

        # 2. Final safety check: Does the chunk *after* chat templating exceed the limit?
        # This might happen if the template adds significant overhead.
        final_token_count = len(tokenizer.encode(prompt))
        if final_token_count > max_input_tokens:
            logging.warning(f"Line {line_number}{shard_info}: Text chunk *after* chat templating still exceeds limit ({final_token_count} > {max_input_tokens}). Skipping this chunk.")
            result_dict["error"] = f"[ERROR: Chunk exceeded limit after templating ({final_token_count} tokens)]"
            # Return error, don't attempt gRPC call
            return result_dict

        # 3. Prepare and send gRPC request for this chunk
        request = jetstream_pb2.DecodeRequest(
            text_content=jetstream_pb2.DecodeRequest.TextContent(text=prompt),
            max_tokens=max_output_tokens,
        )

        if debug:
            logging.debug(f"Line {line_number}{shard_info}: Sending request. Prompt: {final_token_count} tokens.")
            result_dict["prompt"] = prompt # Store prompt only for this chunk in debug

        response_stream = stub.Decode(request, timeout=timeout)
        chunk_response_parts = []
        async for response in response_stream:
            if response and response.stream_content and response.stream_content.samples:
                chunk = response.stream_content.samples[0].text
                if chunk:
                    chunk_response_parts.append(chunk)
            else:
                logging.warning(f"Line {line_number}{shard_info}: Received an unexpected or empty response structure.")

        result_text = "".join(chunk_response_parts).strip()

        # 4. Post-process (e.g., remove end tokens)
        if result_text.endswith("<end_of_turn>"):
            result_text = result_text[:-len("<end_of_turn>")].rstrip()

        result_dict["result"] = result_text
        if debug:
             processing_time = time.monotonic() - start_time
             logging.debug(f"Line {line_number}{shard_info}: Response received ({processing_time:.2f}s): {result_text[:100]}...")
             result_dict["processing_time_seconds"] = round(processing_time, 3)

        return result_dict

    except grpc.aio.AioRpcError as e:
         if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
             logging.error(f"Line {line_number}{shard_info}: Request timed out after {timeout} seconds.")
             result_dict["error"] = f"[TIMEOUT ERROR: Deadline exceeded after {timeout}s]"
         else:
             logging.error(f"Line {line_number}{shard_info}: gRPC Error - Code: {e.code()} Details: {e.details()}")
             result_dict["error"] = f"[gRPC ERROR: {e.code()} - {e.details()}]"
         return result_dict
    except Exception as e:
        logging.error(f"Line {line_number}{shard_info}: Unexpected error processing chunk: {e}", exc_info=True)
        result_dict["error"] = f"[ERROR: {str(e)}]"
        if debug and start_time:
             result_dict["processing_time_seconds"] = round(time.monotonic() - start_time, 3)
        return result_dict


async def process_line_async(
    stub: jetstream_pb2_grpc.OrchestratorStub,
    tokenizer,
    line_number: int,
    line: dict,
    max_output_tokens: int,
    max_input_tokens: int,
    template: str,
    debug: bool,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> Optional[dict]:
    """
    Processes a single line. If input text exceeds token limit, splits into shards
    and processes each shard individually.
    """
    async with semaphore: # Acquire semaphore for the entire line processing (incl. shards)
        line_result = line.copy() # Work on a copy

        try:
            original_text = line.get("text")
            if not original_text:
                logging.warning(f"Line {line_number}: Missing 'text' field. Skipping.")
                line_result["error"] = "[ERROR: Missing 'text' field]"
                return line_result

            # 1. Apply initial template
            templated_text = apply_template(template, original_text)

            # 2. Tokenize *only* to check length against limit
            # Consider executor for tokenization if it becomes CPU bottleneck
            input_tokens = tokenizer.encode(templated_text)
            input_token_count = len(input_tokens)
            del input_tokens # Free memory if large

            # --- Decision Point: Process as single chunk or multiple shards? ---
            if input_token_count <= max_input_tokens:
                # Process as a single chunk (standard case)
                if debug:
                    logging.debug(f"Line {line_number}: Processing as single chunk ({input_token_count} tokens).")
                chunk_result = await _process_single_chunk(
                    text_chunk=templated_text,
                    stub=stub,
                    tokenizer=tokenizer,
                    max_output_tokens=max_output_tokens,
                    max_input_tokens=max_input_tokens,
                    timeout=timeout,
                    debug=debug,
                    line_number=line_number
                )
                if chunk_result.get("error"):
                    line_result["error"] = chunk_result["error"]
                else:
                    line_result["result"] = chunk_result.get("result")
                # Include debug info if present
                if debug:
                    if chunk_result.get("prompt"): line_result["prompt"] = chunk_result.get("prompt")
                    if chunk_result.get("processing_time_seconds"): line_result["processing_time_seconds"] = chunk_result.get("processing_time_seconds")


            else:
                # Process as multiple shards (input exceeds limit)
                safe_shard_limit = int(max_input_tokens * SHARD_TARGET_PERCENTAGE)
                if safe_shard_limit <= 0: # Prevent division by zero or infinite loop
                     logging.error(f"Line {line_number}: Calculated safe_shard_limit is <= 0 ({safe_shard_limit}). Cannot shard. Skipping.")
                     line_result["error"] = "[ERROR: Invalid shard limit calculation]"
                     return line_result

                num_shards = math.ceil(input_token_count / safe_shard_limit)
                logging.info(f"Line {line_number}: Input too long ({input_token_count} tokens > {max_input_tokens}). Splitting into {num_shards} shards.")

                text_shards = split_text_into_shards(templated_text, num_shards)

                shard_results = []
                shard_error_occurred = False
                total_shard_processing_time = 0
                shard_prompts = [] if debug else None # List for shard prompts in debug

                for i, text_shard in enumerate(text_shards):
                    shard_info = f" [Shard {i+1}/{num_shards}]"
                    if not text_shard: # Skip empty shards if split_text produced them
                        logging.debug(f"Line {line_number}{shard_info}: Skipping empty text shard.")
                        continue

                    if debug: logging.debug(f"Line {line_number}{shard_info}: Processing...")

                    chunk_result = await _process_single_chunk(
                        text_chunk=text_shard,
                        stub=stub,
                        tokenizer=tokenizer,
                        max_output_tokens=max_output_tokens,
                        max_input_tokens=max_input_tokens,
                        timeout=timeout,
                        debug=debug,
                        line_number=line_number,
                        shard_info=shard_info
                    )

                    if chunk_result.get("error"):
                        logging.error(f"Line {line_number}{shard_info}: Failed. Error: {chunk_result['error']}")
                        line_result["error"] = f"[ERROR: Shard {i+1}/{num_shards} failed] {chunk_result['error']}"
                        shard_error_occurred = True
                        break # Stop processing shards for this line on first error

                    # Successfully processed shard
                    shard_results.append(chunk_result.get("result", "")) # Append result, default to empty string if missing
                    if debug:
                         if chunk_result.get("prompt"): shard_prompts.append(chunk_result.get("prompt"))
                         total_shard_processing_time += chunk_result.get("processing_time_seconds", 0)

                # After processing all shards (or stopping due to error)
                if not shard_error_occurred:
                    line_result["result"] = SHARD_RESULT_SEPARATOR.join(shard_results)
                    line_result["shards_processed"] = num_shards # Add info about sharding
                    if debug:
                        line_result["processing_time_seconds"] = round(total_shard_processing_time, 3)
                        line_result["prompt"] = shard_prompts # Store list of shard prompts

            return line_result

        except Exception as e:
            # Catch-all for unexpected errors during the line processing logic itself
            logging.error(f"Line {line_number}: Unexpected error processing line: {e}", exc_info=True)
            line_result["error"] = f"[ERROR: {str(e)}]"
            return line_result


async def read_lines_async(filepath: str) -> AsyncGenerator[Tuple[int, Dict], None]:
    """Reads a jsonl file line by line asynchronously, yielding (line_num, data)."""
    try:
        with open(filepath, "r", encoding="utf-8") as infile:
             for line_num, line_str in enumerate(infile, 1):
                try:
                    yield line_num, json.loads(line_str)
                except json.JSONDecodeError:
                    logging.error(f"Skipping malformed JSON at line {line_num}: {line_str.strip()}")
                await asyncio.sleep(0) # Yield control
    except FileNotFoundError:
        logging.error(f"Input file not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading input file {filepath}: {e}")
        raise


async def run(args):
    """Main async function to set up and run the processing pipeline."""
    logging.info(f"Starting processing...")
    logging.info(f"Tokenizer: {args.tokenizer}")
    logging.info(f"Connecting to JetStream server at {args.host}:{args.port}")
    logging.info(f"Max Output Tokens: {args.max_output_tokens}, Max Input Tokens: {args.max_input_tokens}")
    logging.info(f"Concurrent Requests: {args.concurrent_requests}, Request Timeout: {args.timeout}s")
    if args.concurrent_requests > 150:
        logging.warning(f"High concurrency ({args.concurrent_requests}) set. Monitor client/server resources closely.")

    # --- Setup: Load tokenizer and template ---
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
        if not hasattr(tokenizer, 'chat_template') or not tokenizer.chat_template:
            logging.warning(f"Tokenizer {args.tokenizer} does not have a chat_template attribute set. Using basic fallback.")
    except Exception as e:
        logging.error(f"Failed to load tokenizer '{args.tokenizer}': {e}")
        return

    try:
        template = load_template(args.template_file)
    except FileNotFoundError as e:
        logging.error(e)
        return
    except Exception as e:
        logging.error(f"Failed to load template file '{args.template_file}': {e}")
        return

    # --- gRPC Channel and Stub Setup ---
    address = f"{args.host}:{args.port}"
    async with grpc.aio.insecure_channel(address) as channel:
        stub = jetstream_pb2_grpc.OrchestratorStub(channel)

        # Optional: Server reachability check placeholder
        try:
            logging.info(f"Attempting initial connection check to gRPC server at {address}...")
            # Replace with actual health check if available
            logging.info(f"gRPC channel created. Proceeding with requests.")
        except Exception as e:
             logging.error(f"An unexpected error occurred during channel setup/check: {e}")
             return

        # --- Task Preparation and Execution ---
        semaphore = asyncio.Semaphore(args.concurrent_requests)
        tasks = []
        processed_count = 0
        sharded_count = 0 # Count lines that were sharded
        error_count = 0
        skipped_count = 0 # Heuristic/Token skips before processing
        timeout_count = 0
        total_lines = 0

        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                total_lines = sum(1 for _ in f)
            logging.info(f"Found {total_lines} lines in input file {args.input_file}.")
        except FileNotFoundError:
             logging.error(f"Input file {args.input_file} not found.")
             return
        except Exception as e:
             logging.warning(f"Could not determine total lines in input file: {e}")
             total_lines = None

        output_file = None
        try:
            output_file = open(args.output_file, "w", encoding="utf-8")
            async for line_num, line_data in read_lines_async(args.input_file):
                task = asyncio.create_task(process_line_async(
                    stub,
                    tokenizer,
                    line_num,
                    line_data,
                    max_output_tokens=args.max_output_tokens,
                    max_input_tokens=args.max_input_tokens,
                    template=template,
                    debug=args.debug,
                    semaphore=semaphore,
                    timeout=args.timeout
                ))
                tasks.append(task)

            logging.info(f"Submitting {len(tasks)} tasks for processing...")
            for coro in tqdm_asyncio.as_completed(tasks, desc="Processing", total=total_lines or len(tasks)):
                try:
                    processed = await coro
                    if processed:
                        if "error" in processed and processed["error"]:
                             error_count += 1
                             if "TIMEOUT ERROR" in processed["error"]:
                                 timeout_count += 1
                        elif "skipped_reason" in processed:
                             skipped_count += 1
                        else:
                             processed_count += 1
                             # Check if this successful result came from sharding
                             if "shards_processed" in processed:
                                 sharded_count += 1

                        json.dump(processed, output_file, ensure_ascii=False)
                        output_file.write("\n")
                    else:
                         logging.warning("Received unexpected None result from a task.")

                except Exception as e:
                    logging.error(f"Error retrieving result from completed task: {e}", exc_info=True)
                    error_count += 1

        except FileNotFoundError:
             pass # Already logged
        except Exception as e:
             logging.error(f"An error occurred during the main processing loop: {e}", exc_info=True)
        finally:
             if output_file and not output_file.closed:
                 output_file.close()
                 logging.info(f"Output file {args.output_file} closed.")

        # --- Final Summary ---
        logging.info(f"Processing finished.")
        logging.info(f"Results written to {args.output_file}")
        logging.info(f"Successfully processed: {processed_count} (including {sharded_count} sharded lines)")
        logging.info(f"Skipped (heuristic/length before processing): {skipped_count}")
        logging.info(f"Errors: {error_count} (including {timeout_count} timeouts)")
        if timeout_count > error_count * 0.1 and error_count > 5:
             logging.warning("Significant number of timeouts detected. Consider lowering --concurrent_requests or increasing --timeout.")


def main():
    """Parses arguments and runs the async processing."""
    parser = argparse.ArgumentParser(
        description="Asynchronously process text lines via JetStream gRPC, sharding long inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # File paths
    parser.add_argument("--input_file", default="process.jsonl", help="Input .jsonl file path")
    parser.add_argument("--output_file", default="output.jsonl", help="Output .jsonl file path")
    parser.add_argument("--template_file", default="template.txt", help="Template file containing {text}")

    # Connection
    parser.add_argument("--host", default="localhost", help="gRPC server host")
    parser.add_argument("--port", default=9000, type=int, help="gRPC server port")

    # Model/Tokenization parameters
    parser.add_argument("--tokenizer", default="google/gemma-1.1-7b-it", help="HuggingFace tokenizer name/path")
    parser.add_argument("--max_output_tokens", default=4096, type=int, help="Max tokens for generation (output length) *per shard*")
    parser.add_argument("--max_input_tokens", default=4096, type=int, help="Max input tokens per request; inputs longer than this will be sharded")

    # Performance/Control
    parser.add_argument("--concurrent_requests", default=25, type=int, help="Number of concurrent *lines* being processed (each line might involve multiple shard requests)")
    parser.add_argument("--timeout", default=180.0, type=float, help="Timeout in seconds for each individual gRPC shard request")

    # Other
    parser.add_argument("--debug", action="store_true", help="Enable debug logging, include prompt(s) and timing in output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug logging enabled.")
    else:
        logging.getLogger().setLevel(logging.INFO)

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    except Exception as e:
        logging.critical(f"Unhandled exception in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
