const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8080 });
console.log('WebSocket server running on ws://localhost:8080');

let senderSocket = null;
let receiverSocket = null;

wss.on('connection', (ws) => {
  ws.on('message', async (message) => {
    const data = JSON.parse(message);

    if (data.type === 'sender') {
      console.log('Sender connected');
      senderSocket = ws;
      if (receiverSocket) {
        await setupWebRTCConnection();
      }
    } else if (data.type === 'receiver') {
      console.log('Receiver connected');
      receiverSocket = ws;
      if (senderSocket) {
        await setupWebRTCConnection();
      }
    } else if (data.type === 'offer' && ws === receiverSocket) {
      console.log('Offer received from Receiver');
      senderSocket.send(JSON.stringify({ type: 'offer', offer: data.offer }));
    } else if (data.type === 'answer' && ws === senderSocket) {
      console.log('Answer received from Sender');
      receiverSocket.send(JSON.stringify({ type: 'answer', answer: data.answer }));
    } else if (data.type === 'candidate') {
      if (ws === senderSocket && receiverSocket) {
        console.log('Forwarding ICE candidate from Sender to Receiver:', data.candidate); // Add this log
        receiverSocket.send(JSON.stringify({ type: 'candidate', candidate: data.candidate }));
      } else if (ws === receiverSocket && senderSocket) {
        console.log('Forwarding ICE candidate from Receiver to Sender:', data.candidate); // Add this log
        senderSocket.send(JSON.stringify({ type: 'candidate', candidate: data.candidate }));
      }
    }
  });

  ws.on('close', () => {
    if (ws === senderSocket) {
      console.log('Sender disconnected');
      senderSocket = null;
    } else if (ws === receiverSocket) {
      console.log('Receiver disconnected');
      receiverSocket = null;
    }
  });
});

async function setupWebRTCConnection() {
  receiverSocket.send(JSON.stringify({ type: 'start-connection' }));
}
