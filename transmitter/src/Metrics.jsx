// Metrics.js

import React from 'react';

function Metrics({ currentBandwidth, videoResolution, protocol }) {
  return (
    <div style={metricsContainerStyle}>
      <div style={metricStyle}>
        <strong>Bandwidth:</strong> {currentBandwidth} Mbps
      </div>
      <div style={metricStyle}>
        <strong>Resolution:</strong> {videoResolution.width}x{videoResolution.height}
      </div>
      <div style={metricStyle}>
        <strong>Protocol:</strong> {protocol}
      </div>
    </div>
  );
}

// Simple styling for metrics
const metricsContainerStyle = {
  display: 'flex',
  gap: '20px',
  marginTop: '20px',
};

const metricStyle = {
  backgroundColor: '#f0f0f0',
  padding: '10px 15px',
  borderRadius: '5px',
};

export default Metrics;
