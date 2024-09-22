import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import StreamingPlugin from 'chartjs-plugin-streaming';
import 'chartjs-adapter-date-fns';

// Register the necessary components with ChartJS
ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
  StreamingPlugin
);

function BandwidthChart({ bandwidthData }) {
  const data = {
    datasets: [
      {
        label: 'Bandwidth (Mbps)',
        data: bandwidthData.map((value, index) => ({
          x: Date.now() - (bandwidthData.length - index - 1) * 200,
          y: value,
        })),
        borderColor: 'green',
        tension: 0.3,
      },
    ],
  };

  const options = {
    scales: {
      x: {
        type: 'realtime',
        realtime: {
          duration: 60000, // Display data for the last 60 seconds
          refresh: 200, // Refresh every 200 ms
          delay: 200, // Delay of 200 ms
          pause: false,
          frameRate: 30,
        },
        title: {
          display: true,
          text: 'Time',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Bandwidth (Mbps)',
        },
        beginAtZero: true,
      },
    },
    plugins: {
      tooltip: {
        enabled: true,
      },
      legend: {
        display: false,
      },
    },
  };

  return (
    <div>
      <h2>Bandwidth Usage</h2>
      <Line data={data} options={options} />
    </div>
  );
}

export default BandwidthChart;
