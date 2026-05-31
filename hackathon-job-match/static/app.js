const authView = document.querySelector("#authView");
const appShell = document.querySelector("#appShell");
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

renderAuthState();

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
