const authView = document.querySelector("#authView");
const appShell = document.querySelector("#appShell");
const sidebarResizeHandle = document.querySelector("#sidebarResizeHandle");
const loginForm = document.querySelector("#loginForm");
const signupForm = document.querySelector("#signupForm");
const loginEmail = document.querySelector("#loginEmail");
const signupUsername = document.querySelector("#signupUsername");
const authError = document.querySelector("#authError");
const currentUser = document.querySelector("#currentUser");
const logoutButton = document.querySelector("#logoutButton");
const resumeText = document.querySelector("#resumeText");
const resumeFile = document.querySelector("#resumeFile");
const careerUrl = document.querySelector("#careerUrl");
const targetIntent = document.querySelector("#targetIntent");
const voiceButton = document.querySelector("#voiceButton");
const voiceButtonLabel = document.querySelector("#voiceButtonLabel");
const voiceStatus = document.querySelector("#voiceStatus");
const jobsText = document.querySelector("#jobsText");
const analyzeButton = document.querySelector("#analyzeButton");
const useLlm = document.querySelector("#useLlm");
const results = document.querySelector("#results");
const warnings = document.querySelector("#warnings");
const skillsList = document.querySelector("#skillsList");
const memoryPanel = document.querySelector("#memoryPanel");
const memoryBadge = document.querySelector("#memoryBadge");
const memoryStatus = document.querySelector("#memoryStatus");
const memoryContext = document.querySelector("#memoryContext");
const profilePanel = document.querySelector("#profilePanel");
const profileHeadline = document.querySelector("#profileHeadline");
const profileStrengths = document.querySelector("#profileStrengths");
const profileDomains = document.querySelector("#profileDomains");
const profileStrategy = document.querySelector("#profileStrategy");
const skillCount = document.querySelector("#skillCount");
const jobCount = document.querySelector("#jobCount");
const jobTemplate = document.querySelector("#jobTemplate");
let activeUser = loadUser();
let speechRecognition = null;
let isListening = false;
let mediaRecorder = null;
let audioChunks = [];
let voiceMode = "unsupported";

renderAuthState();
setupVoiceInput();
setupSidebarResize();

loginForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const email = loginEmail.value.trim().toLowerCase();
  if (!isValidEmail(email)) {
    showAuthError("Enter a valid email address.");
    return;
  }
  setActiveUser({ id: email, label: email, method: "email" });
});

signupForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const username = normalizeUsername(signupUsername.value);
  if (username.length < 3) {
    showAuthError("Choose a username with at least 3 letters or numbers.");
    return;
  }
  setActiveUser({ id: username, label: username, method: "username" });
});

logoutButton.addEventListener("click", () => {
  localStorage.removeItem("bole_user");
  activeUser = null;
  renderAuthState();
});

resumeFile.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;
  clearWarnings();

  try {
    if (isPdf(file)) {
      resumeText.value = "Extracting text from PDF...";
      resumeText.value = await extractResumeFile(file);
    } else {
      resumeText.value = await file.text();
    }
  } catch (error) {
    resumeText.value = "";
    showWarnings([error.message]);
  }
});

analyzeButton.addEventListener("click", async () => {
  setLoading(true);
  clearWarnings();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: activeUser?.id || "",
        resume_text: resumeText.value,
        career_url: careerUrl.value,
        target_intent: targetIntent.value,
        jobs_text: jobsText.value,
        use_llm: useLlm.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Analysis failed.");
    render(payload);
  } catch (error) {
    showWarnings([error.message]);
  } finally {
    setLoading(false);
  }
});

function setActiveUser(user) {
  activeUser = user;
  localStorage.setItem("bole_user", JSON.stringify(user));
  authError.hidden = true;
  renderAuthState();
}

function loadUser() {
  try {
    const raw = localStorage.getItem("bole_user");
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function renderAuthState() {
  const isSignedIn = Boolean(activeUser?.id);
  authView.hidden = isSignedIn;
  appShell.hidden = !isSignedIn;
  currentUser.textContent = activeUser?.label || "";
}

function setupVoiceInput() {
  if (!voiceButton) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition && navigator.mediaDevices?.getUserMedia && window.MediaRecorder) {
    voiceMode = "recording";
    voiceButton.addEventListener("click", toggleAudioRecording);
    voiceStatus.textContent = "Click to record your target role. Bole will transcribe it with the server.";
    return;
  }

  if (!SpeechRecognition) {
    voiceButton.addEventListener("click", () => {
      targetIntent.focus();
      voiceStatus.textContent = "Voice input is not supported in this browser. Type your target role here.";
    });
    return;
  }

  voiceMode = "speech";
  speechRecognition = new SpeechRecognition();
  speechRecognition.continuous = false;
  speechRecognition.interimResults = true;
  speechRecognition.lang = navigator.language || "en-US";

  speechRecognition.addEventListener("start", () => {
    isListening = true;
    voiceButton.classList.add("is-listening");
    voiceButton.setAttribute("aria-pressed", "true");
    voiceButtonLabel.textContent = "Listening";
    voiceStatus.textContent = "Listening... describe your ideal role.";
  });

  speechRecognition.addEventListener("result", (event) => {
    let finalTranscript = "";
    let interimTranscript = "";

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const transcript = event.results[index][0].transcript.trim();
      if (event.results[index].isFinal) {
        finalTranscript += `${transcript} `;
      } else {
        interimTranscript += `${transcript} `;
      }
    }

    if (interimTranscript) {
      voiceStatus.textContent = interimTranscript.trim();
    }

    if (finalTranscript.trim()) {
      appendTargetIntent(finalTranscript.trim());
    }
  });

  speechRecognition.addEventListener("error", (event) => {
    const message = event.error === "not-allowed"
      ? "Microphone permission was blocked. Allow microphone access or type your target role."
      : "Voice input stopped. You can try again or type your target role.";
    voiceStatus.textContent = message;
  });

  speechRecognition.addEventListener("end", () => {
    isListening = false;
    voiceButton.classList.remove("is-listening");
    voiceButton.setAttribute("aria-pressed", "false");
    voiceButtonLabel.textContent = "Voice input";
    if (!targetIntent.value.trim()) {
      voiceStatus.textContent = "Speak your target role to personalize ranking.";
    }
  });

  voiceButton.addEventListener("click", () => {
    if (isListening) {
      speechRecognition.stop();
      return;
    }
    try {
      speechRecognition.start();
    } catch {
      voiceStatus.textContent = "Voice input is already starting. Try again in a moment.";
    }
  });
}

function setupSidebarResize() {
  if (!appShell || !sidebarResizeHandle) return;

  const savedWidth = Number(localStorage.getItem("bole_sidebar_width") || 0);
  if (savedWidth) {
    setSidebarWidth(savedWidth);
  }

  sidebarResizeHandle.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 900px)").matches) return;
    event.preventDefault();
    sidebarResizeHandle.setPointerCapture(event.pointerId);
    appShell.classList.add("is-resizing");
  });

  sidebarResizeHandle.addEventListener("pointermove", (event) => {
    if (!appShell.classList.contains("is-resizing")) return;
    const bounds = appShell.getBoundingClientRect();
    const width = event.clientX - bounds.left;
    setSidebarWidth(width);
  });

  sidebarResizeHandle.addEventListener("pointerup", (event) => {
    if (!appShell.classList.contains("is-resizing")) return;
    sidebarResizeHandle.releasePointerCapture(event.pointerId);
    appShell.classList.remove("is-resizing");
    localStorage.setItem("bole_sidebar_width", String(getCurrentSidebarWidth()));
  });

  sidebarResizeHandle.addEventListener("pointercancel", () => {
    appShell.classList.remove("is-resizing");
  });

  sidebarResizeHandle.addEventListener("keydown", (event) => {
    if (window.matchMedia("(max-width: 900px)").matches) return;
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    setSidebarWidth(getCurrentSidebarWidth() + direction * 24);
    localStorage.setItem("bole_sidebar_width", String(getCurrentSidebarWidth()));
  });

  sidebarResizeHandle.addEventListener("dblclick", () => {
    localStorage.removeItem("bole_sidebar_width");
    appShell.style.removeProperty("--sidebar-width");
  });
}

function setSidebarWidth(width) {
  const viewport = window.innerWidth || 1200;
  const minWidth = 330;
  const maxWidth = Math.min(820, Math.max(420, viewport - 460));
  const nextWidth = Math.max(minWidth, Math.min(maxWidth, width));
  appShell.style.setProperty("--sidebar-width", `${nextWidth}px`);
}

function getCurrentSidebarWidth() {
  const value = getComputedStyle(appShell).getPropertyValue("--sidebar-width");
  return Math.round(Number.parseFloat(value) || 0);
}

async function toggleAudioRecording() {
  if (isListening && mediaRecorder) {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", async () => {
      stopMicrophoneTracks(stream);
      isListening = false;
      voiceButton.classList.remove("is-listening");
      voiceButton.setAttribute("aria-pressed", "false");
      voiceButtonLabel.textContent = "Voice input";
      await transcribeRecordedAudio();
    });

    mediaRecorder.start();
    isListening = true;
    voiceButton.classList.add("is-listening");
    voiceButton.setAttribute("aria-pressed", "true");
    voiceButtonLabel.textContent = "Stop";
    voiceStatus.textContent = "Recording... click Stop when finished.";
  } catch {
    voiceStatus.textContent = "Microphone permission was blocked. Allow access or type your target role.";
    targetIntent.focus();
  }
}

async function transcribeRecordedAudio() {
  if (!audioChunks.length) {
    voiceStatus.textContent = "No audio was captured. Try again or type your target role.";
    return;
  }

  voiceButton.disabled = true;
  voiceStatus.textContent = "Transcribing your target role...";

  try {
    const audioBlob = new Blob(audioChunks, { type: mediaRecorder?.mimeType || "audio/webm" });
    const response = await fetch("/api/transcribe-voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content_type: audioBlob.type || "audio/webm",
        data: await fileToBase64(audioBlob),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not transcribe voice input.");
    appendTargetIntent(payload.text);
  } catch (error) {
    voiceStatus.textContent = error.message;
  } finally {
    voiceButton.disabled = false;
    audioChunks = [];
    mediaRecorder = null;
  }
}

function stopMicrophoneTracks(stream) {
  stream.getTracks().forEach((track) => track.stop());
}

function appendTargetIntent(transcript) {
  const existing = targetIntent.value.trim();
  targetIntent.value = existing ? `${existing}\n${transcript}` : transcript;
  voiceStatus.textContent = "Added to target role preference.";
  targetIntent.focus();
}

function showAuthError(message) {
  authError.textContent = message;
  authError.hidden = false;
}

function isValidEmail(value) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
}

function normalizeUsername(value) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "")
    .slice(0, 40);
}

function render(payload) {
  renderSkills(payload.resume_skills || []);
  renderMemory(payload.memory);
  renderProfile(payload.candidate_profile);
  jobCount.textContent = payload.job_count || 0;
  showWarnings(payload.warnings || []);

  const recommendations = payload.recommendations || [];
  results.innerHTML = "";

  if (!recommendations.length) {
    results.innerHTML = `
      <div class="empty-board">
        <div class="empty-graphic"></div>
        <h3>No recommendations yet</h3>
        <p>Paste richer job descriptions and run the analysis again.</p>
      </div>
    `;
    return;
  }

  recommendations.forEach((job) => {
    const card = jobTemplate.content.firstElementChild.cloneNode(true);
    card.__job = job;
    const jobUrl = job.url || payload.source_url || "";
    card.querySelector(".score-value").textContent = `${job.score}`;
    const displayScore = job.llm_score || job.score;
    card.querySelector(".score-value").textContent = `${displayScore}`;
    card.querySelector(".score-ring").style.setProperty("--score", `${displayScore}%`);
    card.querySelector(".score-label").textContent = scoreLabel(displayScore);
    card.querySelector(".job-title").textContent = job.title;
    card.querySelector(".why").textContent = job.why;
    card.querySelector(".intent-reason").textContent = job.intent_matches?.length
      ? `Target preference match: ${job.intent_matches.slice(0, 4).join(", ")}`
      : "";
    card.querySelector(".llm-reason").textContent = job.llm_reason || "";
    card.querySelector(".pitch").textContent = job.interview_pitch ? `Pitch: ${job.interview_pitch}` : "";
    renderDistance(card, job, displayScore);
    renderComparison(card, job);

    const link = card.querySelector(".job-link");
    if (jobUrl) {
      link.addEventListener("click", () => {
        window.open(jobUrl, "_blank", "noopener,noreferrer");
      });
    } else {
      link.remove();
    }

    renderChips(card.querySelector(".matched"), job.matched_skills, "No direct overlap");
    renderChips(card.querySelector(".missing"), job.missing_skills, "No obvious gaps");
    renderChips(card.querySelector(".risks"), job.risk_flags, "No concerns found");
    renderGapSummary(card, job);
    renderActionPlan(card, job.recommendation);

    const evidence = card.querySelector(".evidence");
    evidence.innerHTML = (job.evidence || [])
      .map((item) => `<p>${escapeHtml(item)}</p>`)
      .join("");

    card.querySelectorAll("[data-feedback]").forEach((button) => {
      button.addEventListener("click", () => submitJobFeedback(button, card.__job));
    });

    results.appendChild(card);
  });
}

function scoreLabel(score) {
  if (score >= 85) return "Excellent fit";
  if (score >= 70) return "Strong fit";
  if (score >= 55) return "Close match";
  if (score >= 35) return "Stretch";
  return "Low fit";
}

function renderDistance(card, job, score) {
  const missing = job.missing_skills || [];
  const matched = job.matched_skills || [];
  const copy = missing.length
    ? `${100 - score}% distance: strengthen ${missing.slice(0, 3).join(", ")} while keeping ${matched.slice(0, 2).join(", ") || "your strongest evidence"} visible.`
    : `${100 - score}% distance: the JD mostly lines up with your resume evidence. Focus on measurable outcomes.`;
  card.querySelector(".distance-copy").textContent = copy;
  card.querySelector(".distance-meter span").style.width = `${score}%`;
}

function renderGapSummary(card, job) {
  const gaps = job.missing_skills || [];
  const summary = card.querySelector(".gap-summary");
  const severity = gapSeverity(gaps.length);
  summary.innerHTML = `<span class="severity ${severity.className}">${severity.label}</span><span>${gaps.length ? `${gaps.length} gap${gaps.length > 1 ? "s" : ""} to close` : "No critical gaps"}</span>`;
}

function gapSeverity(count) {
  if (count === 0) return { label: "Low", className: "low" };
  if (count <= 2) return { label: "Medium", className: "medium" };
  return { label: "High", className: "high" };
}

function renderComparison(card, job) {
  const matched = job.matched_skills || [];
  const gaps = job.missing_skills || [];
  card.querySelector(".resume-evidence-copy").textContent = matched.length
    ? `Your resume already signals ${matched.slice(0, 4).join(", ")}.`
    : "Bole did not find direct resume evidence for this JD yet.";
  card.querySelector(".jd-requirement-copy").textContent = gaps.length
    ? `The JD also appears to require ${gaps.slice(0, 4).join(", ")}.`
    : "The JD requirements detected by Bole are mostly covered.";
}

function renderActionPlan(card, recommendation) {
  const section = card.querySelector(".action-plan");
  if (!recommendation) {
    section.hidden = true;
    return;
  }

  section.hidden = false;
  section.querySelector("h4").textContent = recommendation.headline || "Recommended next steps";
  section.querySelector(".resume-action").textContent = recommendation.resume_update || "";
  const list = section.querySelector("ul");
  list.innerHTML = "";
  (recommendation.improvement_plan || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

async function submitJobFeedback(button, job) {
  const status = button.parentElement.querySelector(".feedback-status");
  const feedback = button.dataset.feedback;
  const previousText = button.textContent;
  button.disabled = true;
  status.textContent = "Saving...";

  try {
    const response = await fetch("/api/job-feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: activeUser?.id || "",
        feedback,
        job,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not save feedback.");
    status.textContent = payload.memory?.status || "Feedback saved.";
    button.parentElement.querySelectorAll("button").forEach((item) => {
      item.classList.toggle("is-selected", item === button);
    });
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = previousText;
  }
}

function renderMemory(memory) {
  if (!memory) {
    memoryPanel.hidden = true;
    return;
  }

  memoryPanel.hidden = false;
  memoryBadge.textContent = memory.enabled ? "EverOS on" : "Setup needed";
  memoryBadge.classList.toggle("is-muted", !memory.enabled);
  const writeStatus = memory.write?.status ? ` ${memory.write.status}` : "";
  memoryStatus.textContent = `${memory.status || ""}${writeStatus}`.trim();
  memoryContext.innerHTML = "";

  if (!memory.context) {
    memoryContext.hidden = true;
    return;
  }

  memoryContext.hidden = false;
  memory.context
    .split("\n")
    .map((line) => line.replace(/^-\s*/, "").trim())
    .filter(Boolean)
    .forEach((line) => {
      const item = document.createElement("p");
      item.textContent = line;
      memoryContext.appendChild(item);
    });
}

function renderProfile(profile) {
  if (!profile || !profile.headline) {
    profilePanel.hidden = true;
    return;
  }
  profilePanel.hidden = false;
  profileHeadline.textContent = profile.headline;
  profileStrategy.textContent = profile.search_strategy
    ? `Search strategy: ${profile.search_strategy}`
    : "";
  renderChips(profileStrengths, profile.core_strengths || [], "No strengths returned");
  renderChips(profileDomains, profile.domains || [], "No domains returned");
}

function renderSkills(skills) {
  skillCount.textContent = skills.length;
  skillsList.classList.toggle("empty-state", !skills.length);
  renderChips(skillsList, skills, "No skills detected yet");
}

function renderChips(container, items, emptyLabel) {
  container.innerHTML = "";
  if (!items || !items.length) {
    const span = document.createElement("span");
    span.className = "chip muted";
    span.textContent = emptyLabel;
    container.appendChild(span);
    return;
  }
  items.forEach((item) => {
    const span = document.createElement("span");
    span.className = "chip";
    span.textContent = item;
    container.appendChild(span);
  });
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading;
  analyzeButton.querySelector("span").textContent = isLoading ? "Analyzing..." : "Analyze fit";
}

function showWarnings(items) {
  if (!items.length) {
    clearWarnings();
    return;
  }
  warnings.hidden = false;
  warnings.innerHTML = items.map((item) => `<p>${escapeHtml(item)}</p>`).join("");
}

function clearWarnings() {
  warnings.hidden = true;
  warnings.innerHTML = "";
}

async function extractResumeFile(file) {
  const response = await fetch("/api/extract-resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: file.name,
      content_type: file.type,
      data: await fileToBase64(file),
    }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not read this resume file.");
  return payload.text;
}

function isPdf(file) {
  return file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      const dataUrl = String(reader.result || "");
      resolve(dataUrl.split(",")[1] || "");
    });
    reader.addEventListener("error", () => reject(new Error("Could not read this file.")));
    reader.readAsDataURL(file);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
