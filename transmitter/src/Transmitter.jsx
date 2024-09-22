import React, { useRef, useState, useEffect } from 'react';

function Transmitter() {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const peerRef = useRef(null);
  const streamRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  const connectToServer = () => {
    if (isConnected) return;
    const ws = new WebSocket('ws://localhost:8080');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connection opened');
      ws.send(JSON.stringify({ type: 'sender' }));
      setIsConnected(true);
    };

    ws.onmessage = async (message) => {
      const data = JSON.parse(message.data);
      if (data.type === 'offer') {
        console.log('Received offer from server');
        await handleOffer(data.offer);
      } else if (data.type === 'candidate') {
        if (data.candidate) {
          console.log('Received ICE candidate from Receiver');
          await peerRef.current.addIceCandidate(new RTCIceCandidate(data.candidate));
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket connection closed');
      setIsConnected(false);
    };
  };

  const handleOffer = async (offer) => {
    const peer = createPeer();
    peerRef.current = peer; // Make sure to set peerRef.current
    await peer.setRemoteDescription(new RTCSessionDescription(offer));

    // Add local tracks to the peer connection
    streamRef.current.getTracks().forEach((track) => {
      peer.addTrack(track, streamRef.current);
      console.log('Adding track:', track);
    });

    const answer = await peer.createAnswer();
    console.log('SDP Answer:', answer.sdp); // Log the SDP answer
    await peer.setLocalDescription(answer);
    wsRef.current.send(JSON.stringify({ type: 'answer', answer: peer.localDescription }));
  };

  const createPeer = () => {
    const peer = new RTCPeerConnection();

    peer.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('Sending ICE candidate to Receiver:', event.candidate);
        wsRef.current.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
      } else {
        console.log('All ICE candidates have been sent from Sender.');
      }
    };

    return peer;
  };

  const getMedia = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        console.log('Local video stream displayed on sender');
      }
    } catch (err) {
      console.error('Error accessing media devices.', err);
    }
  };

  useEffect(() => {
    getMedia();
  }, []);

  return (
    <div>
      <h1>Transmitter - Sender</h1>
      <div>
        <video ref={videoRef} autoPlay playsInline style={{ width: '600px' }} muted />
      </div>
      <div style={{ marginTop: '20px' }}>
        <button onClick={connectToServer} disabled={isConnected}>
          Connect to Server
        </button>
      </div>
    </div>
  );
}

export default Transmitter;
