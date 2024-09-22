import React, { useRef, useState } from 'react';

function Receiver() {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const peerRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [receivedMessages, setReceivedMessages] = useState([]);

  const connectToServer = () => {
    if (isConnected) return;
    const ws = new WebSocket('ws://localhost:8080');
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connection opened');
      ws.send(JSON.stringify({ type: 'receiver' }));
      setIsConnected(true);
    };

    ws.onmessage = async (message) => {
      const data = JSON.parse(message.data);
      setReceivedMessages((prevMessages) => [
        ...prevMessages,
        JSON.stringify(data, null, 2),
      ]);

      if (data.type === 'start-connection') {
        console.log('Starting WebRTC connection');
        await createPeer();
      } else if (data.type === 'answer') {
        console.log('Received answer from Sender');
        await peerRef.current.setRemoteDescription(
          new RTCSessionDescription(data.answer)
        );
      } else if (data.type === 'candidate') {
        if (data.candidate) {
          console.log('Received ICE candidate from Sender');
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

  const createPeer = async () => {
    const peer = new RTCPeerConnection();
    peerRef.current = peer;

    peer.onicecandidate = (event) => {
      if (event.candidate) {
        console.log('Sending ICE candidate to Sender:', event.candidate);
        wsRef.current.send(JSON.stringify({ type: 'candidate', candidate: event.candidate }));
      } else {
        console.log('All ICE candidates have been sent from Receiver.');
      }
    };

    peer.ontrack = (event) => {
      console.log('Received remote stream from sender');
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
        console.log('Video stream attached to video element');
      }
    };

    // **Add transceivers to specify the media you want to receive**
    peer.addTransceiver('video', { direction: 'recvonly' });
    peer.addTransceiver('audio', { direction: 'recvonly' });

    const offer = await peer.createOffer();
    console.log('SDP Offer:', offer.sdp); // Log the SDP offer
    await peer.setLocalDescription(offer);

    wsRef.current.send(JSON.stringify({ type: 'offer', offer: peer.localDescription }));
  };

  return (
    <div>
      <h1>Receiver</h1>
      <video
        ref={videoRef}
        controls
        autoPlay
        style={{ width: '600px', height: '400px' }}
      ></video>
      <div style={{ marginTop: '20px' }}>
        <button onClick={connectToServer} disabled={isConnected}>
          Connect to Server
        </button>
      </div>

      {/* Display received messages */}
      <div style={{ marginTop: '20px' }}>
        <h2>Received Messages</h2>
        <pre
          style={{
            background: '#f4f4f4',
            padding: '10px',
            maxHeight: '300px',
            overflowY: 'auto',
          }}
        >
          {receivedMessages.length > 0
            ? receivedMessages.join('\n\n')
            : 'No messages received yet.'}
        </pre>
      </div>
    </div>
  );
}

export default Receiver;
