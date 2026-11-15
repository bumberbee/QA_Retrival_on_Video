export function extractionLabel(job, t) {
  if (job?.connectionLost) return t.importReconnecting;
  if (job?.status === "importing") return t.extractStages.import;
  if (job?.status === "extracting") return t.extractStages[job.stage?.split("_")[0]] || t.extracting;
  return t.extractStatuses[job?.status] || t.extracting;
}
