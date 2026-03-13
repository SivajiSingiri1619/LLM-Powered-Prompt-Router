const conversation = document.getElementById("conversation");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const messageTemplate = document.getElementById("messageTemplate");
const heroIntent = document.getElementById("heroIntent");
const heroConfidence = document.getElementById("heroConfidence");
const heroStatus = document.getElementById("heroStatus");
const lastRoute = document.getElementById("lastRoute");

function appendMessage({ role, text, meta = [] }) {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  const metaNode = node.querySelector(".message-meta");
  const bubbleNode = node.querySelector(".bubble");

  node.classList.add(`${role}-message`);
  bubbleNode.textContent = text;

  meta.forEach((item) => {
    const badge = document.createElement("span");
    badge.className = `badge ${item.className}`;
    badge.textContent = item.label;
    metaNode.appendChild(badge);
  });

  conversation.appendChild(node);
  conversation.scrollTop = conversation.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Routing..." : "Route Message";
}

function updateSummary(result) {
  heroIntent.textContent = result.routed_intent;
  heroConfidence.textContent = `${Math.round(result.confidence * 100)}%`;
  heroStatus.textContent = `Classifier chose ${result.intent} and the router answered as ${result.routed_intent}.`;

  const overrideNote = result.manual_override ? "\nManual override: enabled" : "";
  lastRoute.classList.remove("empty");
  lastRoute.textContent =
    `Intent: ${result.intent}\n` +
    `Confidence: ${result.confidence}\n` +
    `Routed as: ${result.routed_intent}` +
    `${overrideNote}\n` +
    `Timestamp: ${result.timestamp}`;
}

async function sendMessage(message) {
  appendMessage({
    role: "user",
    text: message,
    meta: [{ label: "you", className: "badge-user" }],
  });

  setBusy(true);
  heroStatus.textContent = "Routing your request through the classifier and specialist prompt.";

  try {
    const response = await fetch("/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Request failed");
    }

    const meta = [
      { label: `intent: ${payload.intent}`, className: "badge-route" },
      { label: `route: ${payload.routed_intent}`, className: "badge-route" },
      {
        label: `${Math.round(payload.confidence * 100)}% confidence`,
        className: "badge-route",
      },
    ];

    if (payload.manual_override) {
      meta.push({ label: "override", className: "badge-override" });
    }

    appendMessage({
      role: "assistant",
      text: payload.final_response,
      meta,
    });

    updateSummary(payload);
  } catch (error) {
    appendMessage({
      role: "system",
      text: `Request failed: ${error.message}`,
      meta: [{ label: "error", className: "badge-system" }],
    });
    heroIntent.textContent = "error";
    heroConfidence.textContent = "--";
    heroStatus.textContent = "The request did not complete. Check your API key, model settings, and server logs.";
  } finally {
    setBusy(false);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  messageInput.value = "";
  await sendMessage(message);
});

document.querySelectorAll(".quick-tag").forEach((button) => {
  button.addEventListener("click", async () => {
    const message = button.dataset.message || "";
    messageInput.value = "";
    await sendMessage(message);
  });
});
