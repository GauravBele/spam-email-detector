require("dotenv").config();

const cors = require("cors");
const express = require("express");
const mongoose = require("mongoose");
const detectRoutes = require("./routes/detect");

const app = express();
const port = process.env.PORT || 4000;

global.mongoConnected = false;

app.use(cors());
app.use(express.json({ limit: "1mb" }));
app.use("/api", detectRoutes);

app.get("/", (_req, res) => {
  res.json({
    name: "Spam Email Detector API",
    status: "running",
    routes: ["/api/health", "/api/detect", "/api/batch", "/api/history", "/api/stats"],
  });
});

app.use((error, _req, res, _next) => {
  const status = error.response?.status || 500;
  const message =
    error.response?.data?.error || error.message || "Something went wrong.";

  res.status(status).json({ error: message });
});

async function startServer() {
  const mongoUri = process.env.MONGODB_URI;

  if (mongoUri) {
    try {
      await mongoose.connect(mongoUri);
      global.mongoConnected = true;
      console.log("Connected to MongoDB");
    } catch (error) {
      console.warn(`MongoDB unavailable, using in-memory history: ${error.message}`);
    }
  } else {
    console.warn("MONGODB_URI not set, using in-memory history.");
  }

  app.listen(port, () => {
    console.log(`Backend API running on http://localhost:${port}`);
  });
}

startServer();
