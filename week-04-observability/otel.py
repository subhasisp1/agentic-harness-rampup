import time


def export(tracer, exporter=None):
    from opentelemetry import trace as otel
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
    otel_tracer = provider.get_tracer("toolbelt")
    live = {}
    for span in tracer.ordered():  # parents open before their children
        parent = live.get(span.parent_id)
        live[span.span_id] = otel_tracer.start_span(
            span.name,
            context=otel.set_span_in_context(parent) if parent else None,
            start_time=span.start_ns,
            attributes={
                "toolbelt.kind": span.kind,
                "toolbelt.input_tokens": span.input_tokens,
                "toolbelt.output_tokens": span.output_tokens,
                "toolbelt.cost_usd": span.cost,
            },
        )
    for span in sorted(tracer.ordered(), key=lambda s: s.end_ns or 0):  # children end first
        live[span.span_id].end(end_time=span.end_ns or time.time_ns())
    provider.shutdown()
