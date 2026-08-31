function watchRun(runId, onEvent) {
  const source = new EventSource(`/runs/${runId}/events`);
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    onEvent(payload);
    if (payload.status === "done" || payload.status === "error") {
      source.close();
    }
  };
  source.onerror = () => source.close();
}
