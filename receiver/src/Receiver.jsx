import React, { useRef, useState, useEffect } from 'react';
import Metrics from './Metrics'; // Ensure Metrics.js is in the same directory
import './Receiver.css';

function Receiver() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [receivedFrames, setReceivedFrames] = useState(0);
  const [error, setError] = useState(null);
  const [videoResolution, setVideoResolution] = useState({ width: 0, height: 0 });
  const [currentBandwidth, setCurrentBandwidth] = useState(0);

  const bytesReceivedRef = useRef(0);
  const startTimeRef = useRef(null);

  useEffect(() => {
    // Cleanup on component unmount
    return () => {
      disconnectFromServer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Function to connect to WebSocket server
  const connectToServer = () => {
    if (isConnected) {
      console.log('Already connected to the server.');
      return;
    }

    console.log('Attempting to connect to WebSocket server...');
    const ws = new WebSocket('ws://localhost:8080'); // Update if server is on a different host/port
    ws.binaryType = 'arraybuffer'; // Set WebSocket binary type for binary frame handling

    ws.onopen = () => {
      console.log('WebSocket connection opened in Receiver');
      // Send identification message
      const identificationMessage = { type: 'receiver' };
      ws.send(JSON.stringify(identificationMessage));
      console.log('Identification message sent:', identificationMessage);
      setIsConnected(true);
      setError(null);
      bytesReceivedRef.current = 0;
      startTimeRef.current = performance.now();
    };

    ws.onmessage = (message) => {
      if (message.data instanceof ArrayBuffer) {
        console.log('Received binary frame as ArrayBuffer');
        const blob = new Blob([message.data], { type: 'image/jpeg' }); // Assuming JPEG image data

        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // Update video resolution
          setVideoResolution({
            width: img.width || 640,
            height: img.height || 480,
          });
        };
        img.src = URL.createObjectURL(blob);

        // Update received frame count
        setReceivedFrames((prev) => prev + 1);

        // Bandwidth metrics calculation
        bytesReceivedRef.current += message.data.byteLength;
        const currentTime = performance.now();
        const elapsedTime = currentTime - startTimeRef.current; // in milliseconds

        if (elapsedTime > 1000) {
          const bandwidthMbps = (bytesReceivedRef.current * 8) / (elapsedTime * 1000); // Convert to Mbps
          setCurrentBandwidth(bandwidthMbps.toFixed(2));

          // Reset counters
          bytesReceivedRef.current = 0;
          startTimeRef.current = currentTime;
        }
      } else if (typeof message.data === 'string') {
        // Handle text messages (e.g., error messages from server)
        try {
          const data = JSON.parse(message.data);
          if (data.type === 'error') {
            console.error('Server error:', data.message);
            setError(`Server error: ${data.message}`);
          } else {
            console.log('Received text message:', data);
          }
        } catch (err) {
          console.log('Received non-JSON text message:', message.data);
        }
      }
    };

    ws.onerror = (errorEvent) => {
      console.error('WebSocket error in Receiver:', errorEvent);
      setError('WebSocket encountered an error.');
    };

    ws.onclose = (event) => {
      console.log('WebSocket connection closed in Receiver:', event);
      setIsConnected(false);
      setError(null);
    };

    wsRef.current = ws;
  };

  // Function to disconnect from WebSocket server
  const disconnectFromServer = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      console.log('WebSocket connection closed');
    }
    setIsConnected(false);
  };

  return (
    <div className="receiver-container">
      <h1 className="receiver-title">Receiver (Raw Data)</h1>
      <div className="canvas-container">
        <canvas ref={canvasRef} />
      </div>
      <Metrics
        currentBandwidth={currentBandwidth}
        videoResolution={videoResolution}
        protocol="WebSocket"
      />
      {error && <p className="error-message">{error}</p>}
      <div className="receiver-buttons">
        <button onClick={connectToServer} disabled={isConnected}>
          Connect to Server
        </button>
        <button onClick={disconnectFromServer} disabled={!isConnected}>
          Disconnect
        </button>
      </div>
      <div className="metrics">
        <p>Received Frames: {receivedFrames}</p>
      </div>
    </div>
  );
}

export default Receiver;
