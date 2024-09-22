import React from 'react';

function Metrics({ currentBandwidth, videoResolution, protocol }) {
  return (
    <div>
      <h3>Current Bandwidth: {currentBandwidth.toFixed(2)} Mbps</h3>
      <h3>Resolution: {videoResolution.width} x {videoResolution.height}</h3>
      <h3>Protocol: {protocol}</h3>
    </div>
  );
}

export default Metrics;
