export class VoiceChatManager {
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  public isRecording = false;
  public isMuted = true;
  private onAudioPacket?: (base64Audio: string) => void;

  constructor(onAudioPacket?: (base64Audio: string) => void) {
    this.onAudioPacket = onAudioPacket;
  }

  public async startMicrophone(): Promise<boolean> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      this.mediaRecorder = new MediaRecorder(this.mediaStream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size > 0 && !this.isMuted && this.onAudioPacket) {
          const buffer = await e.data.arrayBuffer();
          const base64 = btoa(
            new Uint8Array(buffer).reduce((data, byte) => data + String.fromCharCode(byte), '')
          );
          this.onAudioPacket(base64);
        }
      };

      // Broadcast audio chunks every 300ms for low latency peer voice
      this.mediaRecorder.start(300);
      this.isRecording = true;
      this.isMuted = false;
      return true;
    } catch (err) {
      console.warn('Microphone access denied or unavailable:', err);
      return false;
    }
  }

  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    if (this.mediaStream) {
      this.mediaStream.getAudioTracks().forEach(track => {
        track.enabled = !this.isMuted;
      });
    }
    return this.isMuted;
  }

  public stop() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
    }
    this.isRecording = false;
    this.isMuted = true;
  }
}
