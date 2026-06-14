// Fetch wrappers + job polling for the interpkit GUI API.

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) {
    const detail = data && data.detail !== undefined ? data.detail : res.statusText;
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return data;
}

export const getJSON = (url) => request('GET', url);
export const postJSON = (url, body) => request('POST', url, body);
export const del = (url) => request('DELETE', url);

/**
 * Poll a job until it settles. Calls onUpdate with each snapshot
 * (including the final one). Resolves with the final job; never rejects
 * for op failures — job.status carries the outcome.
 */
export async function pollJob(jobId, { onUpdate = null, interval = 500, maxInterval = 2000 } = {}) {
  let wait = interval;
  for (;;) {
    const job = await getJSON(`/api/jobs/${jobId}`);
    if (onUpdate) onUpdate(job);
    if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled') {
      return job;
    }
    await new Promise((resolve) => setTimeout(resolve, wait));
    wait = Math.min(wait * 1.2, maxInterval);
  }
}

export const cancelJob = (jobId) => del(`/api/jobs/${jobId}`);
