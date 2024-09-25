import React, { useRef, useState, useEffect } from 'react';
import Metrics from './Metrics'; // Ensure Metrics.js is in the same directory
import './Transmitter.css';

function Transmitter() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isAcknowledged, setIsAcknowledged] = useState(false);
  const [videoResolution, setVideoResolution] = useState({ width: 0, height: 0 });
  const [currentBandwidth, setCurrentBandwidth] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const [acknowledgmentStatus, setAcknowledgmentStatus] = useState('none');
  const bytesSentRef = useRef(0);
  const startTimeRef = useRef(null);
  const streamingRef = useRef(false);  // Used to control streaming state
  const [isPaused, setIsPaused] = useState(false); // State to manage pause/resume

  useEffect(() => {
    if (videoRef.current) {
      const video = videoRef.current;
      video.onloadedmetadata = () => {
        setVideoResolution({
          width: video.videoWidth || 640,
          height: video.videoHeight || 480,
        });
        console.log(`Video metadata loaded: ${video.videoWidth}x${video.videoHeight}`);
      };
    }

    // Cleanup on component unmount
    return () => {
      stopMedia();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connectToServer = () => {
    if (isConnected) {
      console.log('Already connected to the server.');
      return;
    }

    console.log('Attempting to connect to WebSocket server...');
    const ws = new WebSocket('ws://localhost:8080');

    ws.onopen = () => {
      console.log('WebSocket connection opened');
      const identificationMessage = { type: 'sender' };
      ws.send(JSON.stringify(identificationMessage));
      console.log('Identification message sent:', identificationMessage);
      setIsConnected(true);
      setError(null);
      setAcknowledgmentStatus('none');
    };

    ws.onmessage = (message) => {
      if (typeof message.data === 'string') {
        try {
          const data = JSON.parse(message.data);
          console.log('Transmitter received message:', data);
          if (data.type === 'sender_acknowledged') {
            console.log('Server acknowledged as sender');
            setIsAcknowledged(true);
            setAcknowledgmentStatus('confirmed');
          } else if (data.type === 'error') {
            console.error('Server error:', data.message);
            setError(`Server error: ${data.message}`);
            setAcknowledgmentStatus('denied');
          }
        } catch (err) {
          console.log('Received non-JSON message:', message.data);
        }
      }
    };

    ws.onerror = (errorEvent) => {
      console.error('WebSocket error:', errorEvent);
      setError('WebSocket encountered an error.');
    };

    ws.onclose = (event) => {
      console.log('WebSocket connection closed:', event);
      setIsConnected(false);
      setIsAcknowledged(false);
      stopStreamingRawData();
      setError(null);
      setAcknowledgmentStatus('none');
    };

    wsRef.current = ws;
  };

  const getMedia = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        console.log('Camera stream started');
      }

      setIsStreaming(true);
    } catch (err) {
      console.error('Error accessing media devices.', err);
      setError('Error accessing camera. Please allow camera access.');
    }
  };

  const stopMedia = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      console.log('WebSocket connection closed');
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      console.log('Camera stream stopped');
    }

    setIsStreaming(false);
    setIsConnected(false);
    setIsAcknowledged(false);
    stopStreamingRawData();
  };

  const startStreamingRawData = () => {
    if (!isConnected || !isAcknowledged || !isStreaming) {
      console.log('Unable to start streaming. Check connection and acknowledgment.');
      return;
    }

    console.log('Starting video stream...');
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const ctx = canvas.getContext('2d');
    let lastFrameTime = 0;
    const frameRate = 60;

    bytesSentRef.current = 0;
    startTimeRef.current = performance.now();
    streamingRef.current = true;

    const streamVideoFrames = () => {
      if (!streamingRef.current) return; // Stop frame streaming if paused or stopped

      const currentTime = performance.now();
      if (currentTime - lastFrameTime >= 1000 / frameRate) {
        if (video) {  // Always check video before streaming frames
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((blob) => {
            if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
              wsRef.current.send(blob);
              console.log('Frame sent to server');

              bytesSentRef.current += blob.size;
              const elapsedTime = currentTime - startTimeRef.current;

              if (elapsedTime > 1000) {
                const bandwidthMbps = (bytesSentRef.current * 8) / (elapsedTime * 1000);
                setCurrentBandwidth(bandwidthMbps.toFixed(2));
                bytesSentRef.current = 0;
                startTimeRef.current = currentTime;
              }
            } else {
              console.log('WebSocket is not open. Unable to send frame.');
            }
          }, 'image/jpeg', 0.7);
          lastFrameTime = currentTime;
        }
      }
      requestAnimationFrame(streamVideoFrames);
    };

    requestAnimationFrame(streamVideoFrames);
  };

  const stopStreamingRawData = () => {
    streamingRef.current = false;  // Stop the stream
    console.log('Stopped streaming video frames');
  };

  const togglePauseResume = () => {
    if (isPaused) {
      // Resuming stream
      console.log('Resuming stream...');
      streamingRef.current = true; // Restart streaming
      startStreamingRawData();  // Restart the streaming loop
    } else {
      // Pausing stream
      console.log('Pausing stream...');
      streamingRef.current = false;  // Stop the streaming
    }
    setIsPaused((prev) => !prev);  // Toggle pause state
  };

  const getNotificationStyle = () => {
    switch (acknowledgmentStatus) {
      case 'confirmed':
        return { backgroundColor: '#4CAF50', color: '#fff' };
      case 'denied':
        return { backgroundColor: '#F44336', color: '#fff' };
      default:
        return { backgroundColor: '#ccc', color: '#666' };
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Transmitter (Raw Data)</h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: '600px', border: '1px solid #ccc' }}
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480" />
      </div>
      <Metrics
        currentBandwidth={currentBandwidth}
        videoResolution={videoResolution}
        protocol="WebSocket"
      />
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <div className="transmitter-buttons">
        <button onClick={connectToServer} disabled={isConnected}>
          Connect to Server
        </button>
        <button onClick={getMedia} disabled={isStreaming || !isConnected}>
          Start Camera
        </button>
        <button onClick={startStreamingRawData} disabled={!isConnected || !isAcknowledged || !isStreaming}>
          Start Streaming
        </button>
        <button onClick={stopMedia} disabled={!isStreaming || !isAcknowledged || !isConnected}>
          Stop Stream
        </button>
        <button onClick={togglePauseResume} disabled={!isStreaming || !isAcknowledged || !isConnected}>
          {isPaused ? 'Resume Stream' : 'Pause Stream'}
        </button>
      </div>

      {/* <div style={{ marginTop: '20px', padding: '10px 20px', borderRadius: '5px', ...getNotificationStyle() }}>
        {acknowledgmentStatus === 'confirmed' && 'Connection confirmed!'}
        {acknowledgmentStatus === 'denied' && 'Connection denied by server.'}
        {acknowledgmentStatus === 'none' && 'Waiting for server response...'}
      </div> */}
    </div>
  );
}

const buttonStyle = {
  padding: '10px 20px',
  fontSize: '16px',
  cursor: 'pointer',
  border: 'none',
  borderRadius: '5px',
};

export default Transmitter;
