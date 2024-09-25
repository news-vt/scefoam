const WebSocket = require('ws');

const PORT = 8080;
const wss = new WebSocket.Server({ port: PORT });

let transmitter = null;
const receivers = new Set();

wss.on('connection', (ws) => {
  ws.id = generateUniqueID();
  ws.role = null;

  console.log('New client connected with ID:', ws.id);

  ws.on('message', (message, isBinary) => {
    if (isBinary) {
      // console.log(`Received binary message from client ${ws.id} with role: ${ws.role}`);
      if (ws.role === 'transmitter') {
        // console.log(`Received binary data from transmitter ${ws.id}. Relaying to receivers.`);
        receivers.forEach((receiver) => {
          if (receiver.readyState === WebSocket.OPEN) {
            receiver.send(message, { binary: true });
          }
        });
      } else {
        console.warn(`Received binary data from non-transmitter client ${ws.id}. Ignoring.`);
        ws.send(JSON.stringify({ type: 'error', message: 'Only transmitter can send binary data.' }));
      }
    } else {
      // Handle text messages
      console.log(`Received text message from client ${ws.id}: ${message}`);

      try {
        const data = JSON.parse(message);
        console.log(`Parsed message from client ${ws.id}:`, data);

        if (data.type === 'sender') {
          if (transmitter) {
            console.log(`Client ${ws.id} attempted to register as transmitter, but transmitter ${transmitter.id} is already connected.`);
            ws.send(JSON.stringify({ type: 'error', message: 'A transmitter is already connected.' }));
            ws.close();
            return;
          }
          transmitter = ws;
          ws.role = 'transmitter';
          console.log(`Client ${ws.id} registered as transmitter.`);
          ws.send(JSON.stringify({ type: 'sender_acknowledged' }));
          console.log(`Sent sender_acknowledged to client ${ws.id}`);
        } else if (data.type === 'receiver') {
          ws.role = 'receiver';
          receivers.add(ws);
          console.log(`Client ${ws.id} registered as receiver.`);
          ws.send(JSON.stringify({ type: 'receiver_acknowledged' }));
        } else {
          console.warn(`Client ${ws.id} sent unknown message type: ${data.type}`);
          ws.send(JSON.stringify({ type: 'error', message: 'Unknown message type.' }));
        }
      } catch (err) {
        console.error(`Error parsing message from client ${ws.id}:`, err);
        ws.send(JSON.stringify({ type: 'error', message: 'Invalid JSON format.' }));
      }
    }
  });

  ws.on('close', () => {
    console.log(`Client ${ws.id} disconnected.`);
    if (ws.role === 'transmitter') {
      console.log(`Transmitter ${ws.id} has disconnected.`);
      transmitter = null;
      // Notify all receivers that the transmitter has disconnected
      receivers.forEach((receiver) => {
        if (receiver.readyState === WebSocket.OPEN) {
          receiver.send(JSON.stringify({ type: 'error', message: 'Transmitter has disconnected.' }));
        }
      });
    } else if (ws.role === 'receiver') {
      receivers.delete(ws);
      console.log(`Receiver ${ws.id} has been removed.`);
    }
  });

  ws.on('error', (error) => {
    console.error(`WebSocket error with client ${ws.id}:`, error);
  });
});

// Utility function to generate unique client IDs
function generateUniqueID() {
  return 'xxxx-xxxx-xxxx'.replace(/[x]/g, () => {
    return Math.floor(Math.random() * 16).toString(16);
  });
}

console.log(`WebSocket server is running on ws://localhost:${PORT}`);
