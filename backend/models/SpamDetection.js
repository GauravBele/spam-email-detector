const mongoose = require("mongoose");

const spamDetectionSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      trim: true,
    },
    label: {
      type: String,
      enum: ["ham", "spam"],
      required: true,
    },
    isSpam: {
      type: Boolean,
      required: true,
    },
    confidence: {
      type: Number,
      required: true,
    },
    spamProbability: {
      type: Number,
      required: true,
    },
    source: {
      type: String,
      enum: ["single", "batch"],
      default: "single",
    },
  },
  { timestamps: true }
);

module.exports = mongoose.model("SpamDetection", spamDetectionSchema);