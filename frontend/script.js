const camera = document.getElementById("camera");
const toggleButton = document.getElementById("toggleCamera");
const delaySelect = document.getElementById("delay");
const poseCanvas = document.getElementById("poseCanvas");
const poseContext = poseCanvas.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");

let stream = null;
let analysisTimer = null;
let delaySeconds = 0;
let isAnalyzing = false;

const poseConnections = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
];

function drawPose(data) {
  poseContext.fillStyle = "#09111f";
  poseContext.fillRect(0, 0, poseCanvas.width, poseCanvas.height);

  poseContext.strokeStyle = "rgba(148, 163, 184, 0.12)";
  poseContext.lineWidth = 1;
  for (let position = 0; position <= 640; position += 40) {
    poseContext.beginPath();
    poseContext.moveTo(position, 0);
    poseContext.lineTo(position, 640);
    poseContext.stroke();
    poseContext.beginPath();
    poseContext.moveTo(0, position);
    poseContext.lineTo(640, position);
    poseContext.stroke();
  }

  if (!data.people || data.people.length === 0) {
    poseContext.fillStyle = "#94a3b8";
    poseContext.font = "18px Segoe UI";
    poseContext.textAlign = "center";
    poseContext.fillText("Không phát hiện người", 320, 320);
    return;
  }

  const scaleX = poseCanvas.width / data.width;
  const scaleY = poseCanvas.height / data.height;

  data.people.forEach((person, personIndex) => {
    poseContext.strokeStyle = personIndex === 0 ? "#20f7b8" : "#60a5fa";
    poseContext.fillStyle = "#ffd166";
    poseContext.shadowColor = "rgba(32, 247, 184, 0.55)";
    poseContext.shadowBlur = 8;
    poseContext.lineWidth = 10;

    poseConnections.forEach(([start, end]) => {
      const pointA = person.keypoints[start];
      const pointB = person.keypoints[end];
      if (!pointA || !pointB || pointA.score < 0.3 || pointB.score < 0.3) return;
      poseContext.beginPath();
      poseContext.moveTo(pointA.x * scaleX, pointA.y * scaleY);
      poseContext.lineTo(pointB.x * scaleX, pointB.y * scaleY);
      poseContext.stroke();
    });

    person.keypoints.forEach((point) => {
      if (point.score < 0.3) return;
      poseContext.beginPath();
      poseContext.arc(point.x * scaleX, point.y * scaleY, 11, 0, Math.PI * 2);
      poseContext.fill();
    });

    poseContext.fillStyle = "rgba(5, 7, 10, 0.82)";
    poseContext.fillRect(10, 10 + personIndex * 38, 250, 32);
    poseContext.fillStyle = "#ffffff";
    poseContext.font = "bold 20px Segoe UI";
    poseContext.textAlign = "left";
    poseContext.fillText(`Khoảng cách: ${person.distance.toFixed(2)} m`, 18, 34 + personIndex * 38);
  });

  if (data.hands) {
    poseContext.strokeStyle = "#ffb04a";
    poseContext.fillStyle = "#ffffff";
    poseContext.shadowColor = "rgba(255, 176, 74, 0.7)";
    poseContext.shadowBlur = 7;
    poseContext.lineWidth = 7;
    data.hands.forEach((hand) => {
      hand.connections.forEach(([start, end]) => {
        const pointA = hand.points[start];
        const pointB = hand.points[end];
        if (!pointA || !pointB) return;
        poseContext.beginPath();
        poseContext.moveTo(pointA.x * scaleX, pointA.y * scaleY);
        poseContext.lineTo(pointB.x * scaleX, pointB.y * scaleY);
        poseContext.stroke();
      });
      hand.points.forEach((point) => {
        poseContext.beginPath();
        poseContext.arc(point.x * scaleX, point.y * scaleY, 6, 0, Math.PI * 2);
        poseContext.fill();
      });

      const wrist = hand.points[0];
      if (wrist) {
        const label = `Tay ${hand.label === "Right" ? "phải" : "trái"}`;
        const labelX = wrist.x * scaleX + 8;
        const labelY = wrist.y * scaleY;
        poseContext.fillStyle = "rgba(5, 7, 10, 0.88)";
        poseContext.fillRect(labelX - 4, labelY - 24, 150, 32);
        poseContext.fillStyle = "#ffffff";
        poseContext.font = "bold 20px Segoe UI";
        poseContext.textAlign = "left";
        poseContext.fillText(label, labelX, labelY);
      }
    });
  }
}

async function analyzeFrame() {
  if (!stream || isAnalyzing || camera.readyState < 2) return;

  isAnalyzing = true;
  captureCanvas.width = camera.videoWidth;
  captureCanvas.height = camera.videoHeight;
  captureContext.save();
  captureContext.translate(captureCanvas.width, 0);
  captureContext.scale(-1, 1);
  captureContext.drawImage(camera, 0, 0, captureCanvas.width, captureCanvas.height);
  captureContext.restore();

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: captureCanvas.toDataURL("image/jpeg", 0.75).split(",")[1] })
    });
    if (response.ok) drawPose(await response.json());
  } catch (error) {
    console.error("Không thể phân tích camera:", error);
  } finally {
    isAnalyzing = false;
  }
}

function startAnalysis() {
  clearInterval(analysisTimer);
  analysisTimer = setInterval(analyzeFrame, 100 + delaySeconds * 1000);
}

toggleButton.addEventListener("click", async () => {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    camera.srcObject = null;
    clearInterval(analysisTimer);
    toggleButton.textContent = "Bật camera";
    poseContext.fillStyle = "#05070a";
    poseContext.fillRect(0, 0, poseCanvas.width, poseCanvas.height);
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false
    });
    camera.srcObject = stream;
    toggleButton.textContent = "Tắt camera";
    startAnalysis();
  } catch (error) {
    alert("Không thể mở camera thiết bị: " + error.message);
  }
});

delaySelect.addEventListener("change", () => {
  delaySeconds = Number(delaySelect.value);
  if (stream) startAnalysis();
});
