// eaclaw-bridge: WhatsApp bridge using whatsmeow.
//
// Communicates with the Rust agent via JSON lines on stdin/stdout.
//
// Inbound (bridge → agent):
//   {"type":"connected"}
//   {"type":"qr","data":"..."}
//   {"type":"message","jid":"...","sender":"...","sender_name":"...","text":"...","timestamp":N,"is_from_me":false}
//
// Outbound (agent → bridge):
//   {"type":"send","jid":"...","text":"..."}

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"go.mau.fi/whatsmeow"
	"go.mau.fi/whatsmeow/proto/waE2E"
	"go.mau.fi/whatsmeow/store/sqlstore"
	"go.mau.fi/whatsmeow/types"
	"go.mau.fi/whatsmeow/types/events"
	waLog "go.mau.fi/whatsmeow/util/log"

	_ "github.com/mattn/go-sqlite3"
	qrcode "github.com/skip2/go-qrcode"
	"google.golang.org/protobuf/proto"
)

// emit writes a JSON line to stdout (→ Rust agent).
func emit(v any) {
	data, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bridge: marshal error: %v\n", err)
		return
	}
	fmt.Println(string(data))
}

// OutboundMsg is a message from the agent asking us to send.
type OutboundMsg struct {
	Type string `json:"type"`
	JID  string `json:"jid"`
	Text string `json:"text"`
}

// renderQR prints a scannable QR code to stderr using Unicode half-blocks.
// Each character encodes two vertical pixels: ▀ ▄ █ and space.
func renderQR(data string) {
	qr, err := qrcode.New(data, qrcode.Medium)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bridge: QR encode error: %v\n", err)
		return
	}
	qr.DisableBorder = false
	bitmap := qr.Bitmap()
	rows := len(bitmap)

	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Scan this QR code with WhatsApp:")
	fmt.Fprintln(os.Stderr, "")

	// Process two rows at a time using half-block characters
	for y := 0; y < rows; y += 2 {
		for x := 0; x < len(bitmap[y]); x++ {
			top := bitmap[y][x]
			bot := false
			if y+1 < rows {
				bot = bitmap[y+1][x]
			}
			// true = black (dark module), false = white
			switch {
			case top && bot:
				fmt.Fprint(os.Stderr, "█")
			case top && !bot:
				fmt.Fprint(os.Stderr, "▀")
			case !top && bot:
				fmt.Fprint(os.Stderr, "▄")
			default:
				fmt.Fprint(os.Stderr, " ")
			}
		}
		fmt.Fprintln(os.Stderr)
	}
	fmt.Fprintln(os.Stderr, "")
}

func main() {
	sessionDir := flag.String("session-dir", "", "directory for WhatsApp session data")
	flag.Parse()

	if *sessionDir == "" {
		home, _ := os.UserHomeDir()
		*sessionDir = home + "/.eaclaw/whatsapp"
	}
	os.MkdirAll(*sessionDir, 0700)

	// SQLite store for whatsmeow device sessions
	ctx := context.Background()
	dbPath := fmt.Sprintf("file:%s/device.db?_foreign_keys=on", *sessionDir)
	container, err := sqlstore.New(ctx, "sqlite3", dbPath, waLog.Noop)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bridge: failed to open store: %v\n", err)
		os.Exit(1)
	}

	device, err := container.GetFirstDevice(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bridge: failed to get device: %v\n", err)
		os.Exit(1)
	}

	client := whatsmeow.NewClient(device, waLog.Noop)

	// Handle incoming events
	client.AddEventHandler(func(evt interface{}) {
		switch v := evt.(type) {
		case *events.Message:
			// Only handle text messages
			msg := v.Message.GetConversation()
			if msg == "" {
				// Try extended text
				if ext := v.Message.GetExtendedTextMessage(); ext != nil {
					msg = ext.GetText()
				}
			}
			if msg == "" {
				return
			}

			senderName := ""
			if v.Info.PushName != "" {
				senderName = v.Info.PushName
			}

			emit(map[string]any{
				"type":        "message",
				"jid":         v.Info.Chat.String(),
				"sender":      v.Info.Sender.String(),
				"sender_name": senderName,
				"text":        msg,
				"timestamp":   v.Info.Timestamp.Unix(),
				"is_from_me":  v.Info.IsFromMe,
			})

		case *events.Connected:
			emit(map[string]string{"type": "connected"})
		}
	})

	// Connect
	if client.Store.ID == nil {
		// No session — need QR code login
		qrChan, _ := client.GetQRChannel(context.Background())
		err = client.Connect()
		if err != nil {
			fmt.Fprintf(os.Stderr, "bridge: connect error: %v\n", err)
			os.Exit(1)
		}

		for evt := range qrChan {
			if evt.Event == "code" {
				// Emit raw data for the Rust agent
				emit(map[string]string{
					"type": "qr",
					"data": evt.Code,
				})
				// Render scannable QR code to stderr
				renderQR(evt.Code)
			}
		}
	} else {
		// Existing session — just connect
		err = client.Connect()
		if err != nil {
			fmt.Fprintf(os.Stderr, "bridge: connect error: %v\n", err)
			os.Exit(1)
		}
	}

	// Read outbound messages from stdin (from Rust agent)
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}

			var msg OutboundMsg
			if err := json.Unmarshal([]byte(line), &msg); err != nil {
				fmt.Fprintf(os.Stderr, "bridge: invalid JSON from agent: %v\n", err)
				continue
			}

			if msg.Type != "send" {
				continue
			}

			jid, err := types.ParseJID(msg.JID)
			if err != nil {
				fmt.Fprintf(os.Stderr, "bridge: invalid JID %q: %v\n", msg.JID, err)
				continue
			}

			_, err = client.SendMessage(context.Background(), jid, &waE2E.Message{
				Conversation: proto.String(msg.Text),
			})
			if err != nil {
				fmt.Fprintf(os.Stderr, "bridge: send error to %s: %v\n", msg.JID, err)
			}
		}
	}()

	// Wait for signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	client.Disconnect()
}
