"""Measure synthetic MCP artifact payloads."""
from __future__ import annotations

import json

from .test_artifacts import BrowserArtifactTests, RUNTIME


def run() -> dict:
    BrowserArtifactTests.setUpClass()
    case = BrowserArtifactTests("test_artifact_scope_and_pause")
    case.setUp()
    try:
        def call(name, **arguments):
            reply = case.server.handle({"jsonrpc":"2.0", "id":"benchmark", "method":"tools/call",
                                        "params":{"name":name, "arguments":arguments}}, mcp_session_key="owner")
            if reply["result"].get("isError"):
                raise AssertionError(reply)
            encoded = json.dumps(reply, ensure_ascii=False)
            return json.loads(reply["result"]["content"][0]["text"]), len(encoded.encode())

        case.load("<main><h1>Document</h1>" + "<p>Rendered report content 汉字.</p>" * 5000 + "</main>")
        _, html_old = call("browser_evaluate", expression="document.body.outerHTML")
        html_artifact, html_new = call("browser_artifact", kind="html", scope_selector="main")
        case.load("<a href='/file.bin'>File attachment</a>")
        snapshot, _ = call("browser_snapshot")
        ref = snapshot["page"]["elements"][0]["ref"]
        _, download_old = call("browser_evaluate", expression="""(async()=>{
            const data=new Uint8Array(await (await fetch(new URL('/file.bin',location.href))).arrayBuffer());
            let binary=''; for(let i=0;i<data.length;i+=8192) binary+=String.fromCharCode(...data.subarray(i,i+8192));
            return btoa(binary);
        })()""")
        download_artifact, download_new = call("browser_artifact", kind="download", ref=ref)
        artifacts = [dict(kind="sanitized_rendered_html", inline_bytes=html_old, artifact_response_bytes=html_new,
                          file_bytes=html_artifact["artifact"]["size_bytes"]),
                     dict(kind="same_origin_download", inline_bytes=download_old, artifact_response_bytes=download_new,
                          file_bytes=download_artifact["artifact"]["size_bytes"])]
        report = dict(measurement="Artifact transfer: UTF-8 bytes of complete MCP replies; later file reads are excluded",
                      artifacts=artifacts,
                      limitations=["Synthetic DOM executing production JS over a Node CDP fixture, not a Chrome conformance test.",
                                   "Cross-origin/redirected download links unsupported; use normal site download flow.",
                                   "Original audit inner outputs may be filtered by the host; these are measured MCP outputs, not attributable real model tokens."])
        path = RUNTIME / "payload-benchmark.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report
    finally:
        case.tearDown()
        case.doCleanups()
        BrowserArtifactTests.tearDownClass()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
