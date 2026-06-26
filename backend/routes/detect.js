const express = require("express");
const axios = require("axios");
const multer = require("multer");
const SpamDetection = require("../models/SpamDetection");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

const historyFallback = [];

function getMlServiceUrl() {
  return process.env.ML_SERVICE_URL || "http://localhost:5000";
}

async function saveDetection(payload) {
  if (global.mongoConnected) {
    return SpamDetection.create(payload);
  }

  const record = {
    _id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    ...payload,
    createdAt: new Date().toISOString(),
  };
  historyFallback.unshift(record);
  return record;
}

async function readHistory(limit) {
  if (global.mongoConnected) {
    return SpamDetection.find().sort({ createdAt: -1 }).limit(limit);
  }

  return historyFallback.slice(0, limit);
}

function parseCsvEmails(buffer) {
  const text = buffer.toString("utf8");
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      if (index === 0 && /^email(,|$)/i.test(line)) {
        return "";
      }
      return line.replace(/^"|"$/g, "");
    })
    .filter(Boolean);
}

router.get("/health", async (_req, res) => {
  try {
    const mlResponse = await axios.get(`${getMlServiceUrl()}/health`, { timeout: 5000 });
    res.json({
      status: "ok",
      database: global.mongoConnected ? "mongodb" : "memory",
      mlService: mlResponse.data,
    });
  } catch (error) {
    res.status(503).json({
      status: "degraded",
      database: global.mongoConnected ? "mongodb" : "memory",
      mlService: "unavailable",
      error: error.message,
    });
  }
});

router.post("/detect", async (req, res, next) => {
  try {
    const email = req.body.email;

    if (!email || typeof email !== "string") {
      return res.status(400).json({ error: "email is required." });
    }

    const mlResponse = await axios.post(`${getMlServiceUrl()}/predict`, { email });
    const detection = await saveDetection({
      email,
      label: mlResponse.data.label,
      isSpam: mlResponse.data.isSpam,
      confidence: mlResponse.data.confidence,
      spamProbability: mlResponse.data.spamProbability,
      source: "single",
    });

    res.json(detection);
  } catch (error) {
    next(error);
  }
});

router.post("/batch", upload.single("file"), async (req, res, next) => {
  try {
    const emails = req.file ? parseCsvEmails(req.file.buffer) : req.body.emails;

    if (!Array.isArray(emails) || emails.length === 0) {
      return res.status(400).json({ error: "Upload a CSV file or send emails as an array." });
    }

    const mlResponse = await axios.post(`${getMlServiceUrl()}/batch_predict`, { emails });
    const records = await Promise.all(
      mlResponse.data.results.map((result) =>
        saveDetection({
          email: result.email,
          label: result.label,
          isSpam: result.isSpam,
          confidence: result.confidence,
          spamProbability: result.spamProbability,
          source: "batch",
        })
      )
    );

    res.json({ count: records.length, results: records });
  } catch (error) {
    next(error);
  }
});

router.get("/history", async (req, res, next) => {
  try {
    const limit = Math.min(Number(req.query.limit) || 25, 100);
    const history = await readHistory(limit);
    res.json(history);
  } catch (error) {
    next(error);
  }
});

router.get("/stats", async (_req, res, next) => {
  try {
    const history = await readHistory(1000);
    const total = history.length;
    const spam = history.filter((item) => item.isSpam).length;
    const ham = total - spam;
    const averageConfidence =
      total === 0
        ? 0
        : history.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / total;

    res.json({
      total,
      spam,
      ham,
      spamRate: total === 0 ? 0 : spam / total,
      averageConfidence,
    });
  } catch (error) {
    next(error);
  }
});

module.exports = router;
