//FrameProcessor.jsx
class FrameProcessor {
    constructor(stream, ws) {
      this.stream = stream;
      this.ws = ws;
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');
      this.video = document.createElement('video');
      this.video.srcObject = stream;
      this.isRunning = false;
  
      this.video.onloadedmetadata = () => {
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.video.play(); // Start video playback
        this.start(); // Start processing frames
      };
    }
  
    start() {
      this.isRunning = true;
      this.processFrame();
    }
  
    stop() {
      this.isRunning = false;
      this.video.pause();
    }
  
    processFrame() {
      if (!this.isRunning) return;
  
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
  
      // Convert the canvas to a Blob and send via WebSocket
      this.canvas.toBlob(
        (blob) => {
          if (blob && this.ws.readyState === WebSocket.OPEN) {
            console.log('Sending frame with blob size:', blob.size); // Log blob size
            this.ws.send(blob); // Send the frame blob
          } else {
            console.log('Frame not sent, WebSocket not open or invalid blob.');
          }
          requestAnimationFrame(this.processFrame.bind(this)); // Continue processing frames
        },
        'image/jpeg',
        0.8 // Adjust JPEG quality if needed
      );
    }
  }
  
  export default FrameProcessor;
  