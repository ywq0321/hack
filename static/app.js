// Segmentation viewer & manipulator frontend
// - Upload image and run segmentation (Flask server handles inference)
// - Load mask PNGs (0/255) and cut out true object regions from base image
// - Click pixel → select *all overlapping masks* covering that pixel
// - Drag → move those objects together (background stays fixed)
// - Press "r" → reset positions

const uploadForm = document.getElementById("uploadForm");
const logEl = document.getElementById("log");
const baseContainer = document.getElementById("baseContainer");
const masksContainer = document.getElementById("masksContainer");

let baseImage = new Image();
let baseCanvas = null;
let baseCtx = null;
let masks = []; // {name,url,maskImg,processedMask,maskCanvas,pieceCanvas,offset:{x,y}}
let selectedMaskIndices = [];
let dragging = false;
let dragStart = null;

function log(...args) {
  logEl.textContent = args.join(" ") + "\n" + logEl.textContent;
}

// ---------- Upload & segmentation ----------
uploadForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const fileInput = document.getElementById("image");
  if (!fileInput.files || fileInput.files.length === 0)
    return alert("Pick an image");

  const form = new FormData();
  form.append("image", fileInput.files[0]);
  form.append("model_path", document.getElementById("model_path").value || "");

  log("Uploading image and running segmentation...");
  const res = await fetch("/segment", { method: "POST", body: form });
  const data = await res.json();
  if (data.error) {
    log("Error: " + data.error);
    return;
  }
  log(
    "Segmentation finished. stdout:\n" +
      (data.stdout || "") +
      "\nstderr:\n" +
      (data.stderr || "")
  );

  await loadBaseImage(data.image_url);
  masks = [];
  masksContainer.innerHTML = "";

  for (let i = 0; i < data.masks.length; i++) {
    const m = data.masks[i];
    await addMask(m.name, m.url);
  }

  log("Loaded " + masks.length + " mask(s)");
});

// ---------- Load base image ----------
function loadBaseImage(url) {
  return new Promise((resolve, reject) => {
    baseImage = new Image();
    baseImage.crossOrigin = "anonymous";
    baseImage.onload = () => {
      baseContainer.innerHTML = "";
      baseCanvas = document.createElement("canvas");
      baseCanvas.className = "baseImage";
      baseCanvas.width = baseImage.naturalWidth;
      baseCanvas.height = baseImage.naturalHeight;
      baseCtx = baseCanvas.getContext("2d");
      baseCtx.drawImage(baseImage, 0, 0);
      baseContainer.appendChild(baseCanvas);

      masksContainer.style.width = baseCanvas.width + "px";
      masksContainer.style.height = baseCanvas.height + "px";
      masksContainer.style.position = "absolute";
      masksContainer.style.left = baseContainer.offsetLeft + "px";
      masksContainer.style.top = baseContainer.offsetTop + "px";

      resolve();
    };
    baseImage.onerror = reject;
    baseImage.src = url;
  });
}

// ---------- Utility ----------
function randomColor() {
  const r = Math.floor(Math.random() * 200) + 30;
  const g = Math.floor(Math.random() * 200) + 30;
  const b = Math.floor(Math.random() * 200) + 30;
  return { r, g, b };
}
function getMousePosOnCanvas(e) {
  const rect = baseCanvas.getBoundingClientRect();
  const x = Math.round(
    (e.clientX - rect.left) * (baseCanvas.width / rect.width)
  );
  const y = Math.round(
    (e.clientY - rect.top) * (baseCanvas.height / rect.height)
  );
  return { x, y };
}

// ---------- Add mask & cutout ----------
async function addMask(name, url) {
  const maskImg = new Image();
  maskImg.crossOrigin = "anonymous";
  await new Promise((res, rej) => {
    maskImg.onload = res;
    maskImg.onerror = rej;
    maskImg.src = url;
  });

  const w = maskImg.naturalWidth,
    h = maskImg.naturalHeight;

  // read mask pixels
  const tmpC = document.createElement("canvas");
  tmpC.width = w;
  tmpC.height = h;
  const tmpCtx = tmpC.getContext("2d");
  tmpCtx.drawImage(maskImg, 0, 0);
  const maskData = tmpCtx.getImageData(0, 0, w, h).data;

  // detect inversion
  let white = 0;
  for (let i = 0; i < maskData.length; i += 4)
    if (maskData[i] > 128) white++;
  const inverted = white / (w * h) > 0.5;

  // produce normalized mask (white = object)
  const procMask = document.createElement("canvas");
  procMask.width = w;
  procMask.height = h;
  const pctx = procMask.getContext("2d");
  const procData = pctx.createImageData(w, h);
  for (let i = 0; i < maskData.length; i += 4) {
    const val = maskData[i];
    const obj = inverted ? val < 128 : val > 128;
    procData.data[i + 0] = procData.data[i + 1] = procData.data[i + 2] =
      obj ? 255 : 0;
    procData.data[i + 3] = obj ? 255 : 0;
  }
  pctx.putImageData(procData, 0, 0);

  // create tinted overlay
  const maskCanvas = document.createElement("canvas");
  maskCanvas.className = "maskCanvas";
  maskCanvas.width = w;
  maskCanvas.height = h;
  maskCanvas.style.position = "absolute";
  maskCanvas.style.left = "0px";
  maskCanvas.style.top = "0px";
  const mctx = maskCanvas.getContext("2d");
  const c = randomColor();
  mctx.fillStyle = `rgba(${c.r},${c.g},${c.b},0.35)`;
  mctx.fillRect(0, 0, w, h);
  mctx.globalCompositeOperation = "destination-in";
  mctx.drawImage(procMask, 0, 0);

  // create pieceCanvas (actual object)
  const pieceCanvas = document.createElement("canvas");
  pieceCanvas.className = "pieceCanvas";
  pieceCanvas.width = w;
  pieceCanvas.height = h;
  pieceCanvas.style.position = "absolute";
  pieceCanvas.style.left = "0px";
  pieceCanvas.style.top = "0px";
  pieceCanvas.style.pointerEvents = "none";
  const pCtx = pieceCanvas.getContext("2d");

  // draw base image then apply mask as alpha
  pCtx.drawImage(baseImage, 0, 0);
  const pieceData = pCtx.getImageData(0, 0, w, h);
  const maskPix = pctx.getImageData(0, 0, w, h).data;
  for (let i = 0; i < pieceData.data.length; i += 4)
    pieceData.data[i + 3] = maskPix[i + 3];
  pCtx.putImageData(pieceData, 0, 0);

  // erase region from base
  baseCtx.save();
  baseCtx.globalCompositeOperation = "destination-out";
  baseCtx.drawImage(procMask, 0, 0);
  baseCtx.restore();

  masksContainer.appendChild(pieceCanvas);
  masksContainer.appendChild(maskCanvas);

  masks.push({
    name,
    url,
    maskImg,
    processedMask: procMask,
    maskCanvas,
    pieceCanvas,
    offset: { x: 0, y: 0 },
  });

  maskCanvas.addEventListener("mousedown", onMaskMouseDown);
  maskCanvas.addEventListener("touchstart", (ev) => {
    ev.preventDefault();
    onMaskMouseDown(ev.touches[0]);
  });
}

// ---------- Selection & dragging ----------
function onMaskMouseDown(e) {
  e.preventDefault();
  const pos = getMousePosOnCanvas(e);
  selectedMaskIndices = [];

  // include all masks that contain the pixel
  masks.forEach((m, idx) => {
    const localX = pos.x - m.offset.x;
    const localY = pos.y - m.offset.y;
    if (
      localX < 0 ||
      localY < 0 ||
      localX >= m.processedMask.width ||
      localY >= m.processedMask.height
    )
      return;
    const ctx = m.processedMask.getContext("2d");
    const pix = ctx.getImageData(localX, localY, 1, 1).data;
    if (pix[3] > 10) selectedMaskIndices.push(idx);
  });

  if (selectedMaskIndices.length === 0) return;

  dragging = true;
  dragStart = { x: e.clientX, y: e.clientY };

  selectedMaskIndices.forEach((i) => {
    masks[i].maskCanvas.style.zIndex = 2000;
    masks[i].pieceCanvas.style.zIndex = 1999;
  });

  window.addEventListener("mousemove", onDrag);
  window.addEventListener("mouseup", onDrop);
  window.addEventListener("touchmove", touchMoveHandler, { passive: false });
  window.addEventListener("touchend", touchEndHandler);
}

function onDrag(e) {
  if (!dragging) return;
  const dx = Math.round(e.clientX - dragStart.x);
  const dy = Math.round(e.clientY - dragStart.y);
  dragStart = { x: e.clientX, y: e.clientY };

  selectedMaskIndices.forEach((i) => {
    const m = masks[i];
    m.offset.x += dx;
    m.offset.y += dy;
    const tr = `translate(${m.offset.x}px, ${m.offset.y}px)`;
    m.maskCanvas.style.transform = tr;
    m.pieceCanvas.style.transform = tr;
  });
}
function onDrop() {
  dragging = false;
  selectedMaskIndices.forEach((i) => {
    masks[i].maskCanvas.style.zIndex = "";
    masks[i].pieceCanvas.style.zIndex = "";
  });
  selectedMaskIndices = [];
  window.removeEventListener("mousemove", onDrag);
  window.removeEventListener("mouseup", onDrop);
  window.removeEventListener("touchmove", touchMoveHandler);
  window.removeEventListener("touchend", touchEndHandler);
}
function touchMoveHandler(ev) {
  ev.preventDefault();
  if (ev.touches && ev.touches[0]) onDrag(ev.touches[0]);
}
function touchEndHandler(ev) {
  if (ev.changedTouches && ev.changedTouches[0]) onDrop(ev.changedTouches[0]);
}

// ---------- Reset ----------
window.addEventListener("keydown", (e) => {
  if (e.key === "r") {
    masks.forEach((m) => {
      m.offset = { x: 0, y: 0 };
      m.maskCanvas.style.transform = "";
      m.pieceCanvas.style.transform = "";
    });
    log("Reset mask positions");
  }
});
