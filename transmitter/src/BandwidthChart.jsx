// src/BandwidthChart.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, LineElement, PointElement, Title, Tooltip, Legend);

function BandwidthChart({ bandwidthData }) {
  const data = {
    labels: bandwidthData.map((_, index) => index + 1),
    datasets: [
      {
        label: 'Bandwidth (bps)',
        data: bandwidthData,
        fill: false,
        borderColor: 'blue',
      },
    ],
  };

  return (
    <div>
      <h2>Bandwidth Usage</h2>
      <Line data={data} />
    </div>
  );
}

export default BandwidthChart;
