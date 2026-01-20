"""
Server-Sent Events (SSE) Streaming for LLM Responses

This module implements streaming responses for real-time text generation.
SSE allows the server to push updates to the client as tokens are generated.

Learning Objectives:
- Understand streaming patterns for LLMs
- Implement SSE (Server-Sent Events) protocol
- Handle backpressure and flow control
- Implement proper error handling in streams
- Learn async iteration patterns

Key Concepts:
- Server-Sent Events (SSE) protocol
- Async generators in Python
- Streaming vs batch generation
- Token-by-token vs chunk streaming
- Error handling in streams

Benefits of Streaming:
- Lower perceived latency (users see results immediately)
- Better user experience for long generations
- Ability to cancel long-running requests
- Progressive rendering of responses
"""

import asyncio
import json
from typing import AsyncIterator, Optional, Dict, Any, List
from fastapi.responses import StreamingResponse
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SSE UTILITIES
# ============================================================================

def format_sse(data: str, event: Optional[str] = None, id: Optional[str] = None) -> str:
    """
    Format data as Server-Sent Event.
    """
    message = ""
    if id is not None:
        message += f"id: {id}\n"
    if event is not None:
        message += f"event: {event}\n"
    
    # Handle multi-line data
    lines = data.split("\n")
    for line in lines:
        message += f"data: {line}\n"
    
    message += "\n"
    return message


def create_sse_response(
    generator: AsyncIterator[str],
    status_code: int = 200
) -> StreamingResponse:
    """
    Create FastAPI StreamingResponse for SSE.
    """
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    }

    return StreamingResponse(
        generator,
        status_code=status_code,
        headers=headers,
        media_type="text/event-stream"
    )


# ============================================================================
# STREAMING GENERATORS
# ============================================================================

async def stream_llm_response(
    llm_generator: AsyncIterator[str],
    request_id: str,
    include_metadata: bool = True
) -> AsyncIterator[str]:
    """
    Stream LLM tokens as SSE events.
    """
    start_time = datetime.now()
    token_count = 0

    try:
        # Send initial event
        yield format_sse(
            json.dumps({"request_id": request_id, "status": "started"}),
            event="start"
        )

        async for token in llm_generator:
            token_count += 1
            
            # Format token as chunk event
            chunk_data = {
                "text": token,
                "token_count": token_count
            }
            yield format_sse(json.dumps(chunk_data), event="chunk")

            # Optionally send periodic metadata
            if include_metadata and token_count > 0 and token_count % 10 == 0:
                 elapsed = (datetime.now() - start_time).total_seconds()
                 yield format_sse(
                     json.dumps({
                         "token_count": token_count,
                         "elapsed_seconds": round(elapsed, 3),
                         "tokens_per_sec": round(token_count / elapsed, 2)
                     }),
                     event="metadata"
                 )

        # Send completion event
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000

        done_data = {
            "status": "complete",
            "token_count": token_count,
            "latency_ms": latency_ms,
            "request_id": request_id
        }
        yield format_sse(json.dumps(done_data), event="done")

    except ValueError as ve: # Catch cancellation or specific logic
        logger.warning(f"Validation error in stream {request_id}: {ve}")
        yield format_sse(json.dumps({"error": str(ve)}), event="error")
        
    except Exception as e:
        logger.error(f"Error in stream {request_id}: {e}", exc_info=True)
        error_data = {
            "error": str(e),
            "request_id": request_id
        }
        yield format_sse(json.dumps(error_data), event="error")


async def stream_rag_response(
    rag_pipeline,
    query: str,
    request_id: str,
    **generation_params
) -> AsyncIterator[str]:
    """
    Stream RAG-augmented generation.
    """
    try:
        # Send start event
        yield format_sse(
            json.dumps({"status": "started", "request_id": request_id}),
            event="start"
        )

        # Send retrieving event
        yield format_sse(
            json.dumps({"status": "retrieving"}),
            event="retrieving"
        )

        # Perform retrieval
        # Assuming rag_pipeline has a retrieve method
        sources = []
        if hasattr(rag_pipeline, "retrieve"):
            sources = await rag_pipeline.retrieve(query)
        else:
            # Mock if not implemented/available for testing
            sources = [] 

        # Send sources event
        sources_data = {
            "sources": [
                {
                    "content": doc.text if hasattr(doc, 'text') else str(doc)[:200],  # Truncate
                    "score": getattr(doc, 'score', 0.0),
                    "metadata": getattr(doc, 'metadata', {})
                }
                for doc in sources
            ],
            "num_sources": len(sources)
        }
        yield format_sse(json.dumps(sources_data), event="sources")

        # Build prompt with context
        prompt = query # Placeholder, normally rag_pipeline.build_prompt(query, sources)
        if hasattr(rag_pipeline, "build_prompt"):
             prompt = rag_pipeline.build_prompt(query, sources)

        # Send generating event
        yield format_sse(
            json.dumps({"status": "generating"}),
            event="generating"
        )

        # Stream generation
        if hasattr(rag_pipeline, "generate_stream"):
            llm_generator = rag_pipeline.generate_stream(prompt, **generation_params)
            async for chunk in llm_generator:
                yield format_sse(
                    json.dumps({"text": chunk}),
                    event="chunk"
                )
        else:
             # Mock generation usually needed
             pass

        # Send done event
        yield format_sse(
            json.dumps({"status": "complete", "request_id": request_id}),
            event="done"
        )

    except Exception as e:
        logger.error(f"RAG stream error {request_id}: {e}")
        yield format_sse(
            json.dumps({"error": str(e)}),
            event="error"
        )


async def stream_with_heartbeat(
    generator: AsyncIterator[str],
    heartbeat_interval: int = 30
) -> AsyncIterator[str]:
    """
    Wrap generator with heartbeat events.
    """
    while True:
        try:
            # Wait for next item with timeout
            # Note: anext(iterator) is python 3.10+, using generator.__anext__() for compat
            item = await asyncio.wait_for(
                generator.__anext__(),
                timeout=heartbeat_interval
            )
            yield item
        except asyncio.TimeoutError:
            # Send heartbeat
            yield format_sse(
                json.dumps({"type": "heartbeat", "timestamp": str(datetime.now())}),
                event="heartbeat"
            )
        except StopAsyncIteration:
            break
        except Exception as e:
            # Propagate other errors
            raise e


# ============================================================================
# BATCH STREAMING
# ============================================================================

async def stream_batch_responses(
    requests: list,
    llm_server,
    max_concurrent: int = 3
) -> AsyncIterator[str]:
    """
    Stream multiple requests concurrently.
    Placeholder / TODO implementation.
    """
    yield format_sse(json.dumps({"status": "Not implemented"}), event="error")


# ============================================================================
# ERROR HANDLING AND RECOVERY
# ============================================================================

async def safe_stream_wrapper(
    generator: AsyncIterator[str],
    request_id: str,
    on_error: Optional[callable] = None
) -> AsyncIterator[str]:
    """
    Wrap stream with error handling and recovery.
    """
    try:
        async for item in generator:
            yield item
    except asyncio.CancelledError:
        logger.info(f"Stream {request_id} cancelled")
        yield format_sse(
            json.dumps({"status": "cancelled"}),
            event="cancelled"
        )
        raise
    except Exception as e:
        logger.error(f"Stream {request_id} error: {e}")
        yield format_sse(
            json.dumps({"error": str(e), "error_type": type(e).__name__}),
            event="error"
        )
        if on_error:
            try:
                await on_error(e)
            except:
                pass


# ============================================================================
# UTILITIES
# ============================================================================

class StreamBuffer:
    """
    Buffer for aggregating tokens before sending.
    """

    def __init__(self, buffer_size: int = 5, flush_timeout: float = 0.1):
        """
        Initialize buffer.
        """
        self.buffer_size = buffer_size
        self.flush_timeout = flush_timeout
        self.buffer: List[str] = []
        self.last_flush = time.time()

    async def add_token(self, token: str) -> Optional[str]:
        """
        Add token to buffer.
        """
        self.buffer.append(token)
        
        should_flush = (
            len(self.buffer) >= self.buffer_size or
            (time.time() - self.last_flush) > self.flush_timeout
        )
        
        if should_flush:
            return await self.flush()
        return None

    async def flush(self) -> str:
        """
        Flush buffer contents.
        """
        if not self.buffer:
            return ""
        
        text = "".join(self.buffer)
        self.buffer = []
        self.last_flush = time.time()
        return text


def calculate_stream_metrics(
    start_time: datetime,
    token_count: int,
    first_token_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculate streaming performance metrics.
    """
    total_latency = (datetime.now() - start_time).total_seconds() * 1000
    ttft_ms = 0.0
    if first_token_time:
        ttft_ms = (first_token_time - start_time).total_seconds() * 1000
        
    tps = 0.0
    if total_latency > 0:
        tps = token_count / (total_latency / 1000.0)

    return {
        "ttft_ms": round(ttft_ms, 2),
        "tokens_per_second": round(tps, 2),
        "total_latency_ms": round(total_latency, 2),
        "total_tokens": token_count
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example FastAPI Endpoint:

from fastapi import FastAPI
from .streaming import stream_llm_response, create_sse_response

app = FastAPI()

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    # Get LLM generator
    llm_generator = llm_server.generate_stream(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Wrap with SSE formatting
    sse_generator = stream_llm_response(
        llm_generator,
        request_id=generate_request_id(),
        include_metadata=True
    )

    # Return streaming response
    return create_sse_response(sse_generator)
"""
