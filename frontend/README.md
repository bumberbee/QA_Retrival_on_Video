# Video Q&A frontend

A React and Vite workspace for importing videos, exploring slides and transcripts, and asking questions about a video's content. The interface supports Vietnamese and English, the original warm amber palette, and saved light/dark preferences (dark by default).

## Development

```powershell
npm install
npm run dev
```

Open `http://localhost:5173`. Start the FastAPI backend on port `8001` as described in the root README, using `--port 8001`.

The API defaults to `http://localhost:8001/api`. To use a different backend, create `frontend/.env.local` and restart Vite:

```dotenv
VITE_API_BASE_URL=http://localhost:8000/api
```

For a deployment with a reverse proxy, this can be `/api`. The backend must allow requests from the frontend's origin.

## Interface

- Search, sort, filter by transcript availability, and switch between grid and list views in the library.
- Open **Ask AI** for a dedicated conversation page: browse videos with transcripts, open their chat directly, or prepare a suggested question. Videos awaiting transcripts are linked back to the library.
- Choose **Paste a link** to import a public YouTube, Facebook, or TikTok video. The dialog shows download progress, failures, and an action to open the finished video. It can be closed while downloading; the status indicator reopens it. Pending imports resume polling after a page refresh in the same tab; temporary connection failures retry automatically. The backend needs yt-dlp and ffmpeg as described in the root README.
- Upload MP4, WebM, MOV, or MKV files. AI questions require a transcript to be added through the existing backend workflow.
- Browse slides, open them with keyboard navigation, or search the transcript and jump to its timestamps.
- Rewind or advance by 10 seconds with the player buttons. Use left/right arrows for 5 seconds or J/L for 10; double-tap either side of the video on touch screens. The speed selector supports 0.25×–2× and remembers your preference.
- Ask questions alongside the video on desktop or in the chat sheet on mobile. Answer timestamps seek the player; slide positions are approximate.
- Press `Ctrl+K` or `Cmd+K` to search the library, and `Escape` to close a dialog.

The player/chat code is loaded when a video is opened. Slide previews load near the viewport, slide metadata is shared between previews and the player, and transcripts load when their tab is opened. Large libraries and transcripts render in batches.

## Checks

```powershell
npm run lint
npm run build
```

`src/App.jsx` manages the application shell and uploads. `src/components/VideoRail.jsx` lists the videos and `src/components/Workspace.jsx` holds the player and Q&A view. `src/components/ImportDialog.jsx` handles the link/file choices, while `src/lib/useVideoImport.js` manages import jobs and polling. Shared controls are in `src/components/UI.jsx`, translations and API helpers are in `src/lib`, and the responsive styles are in `src/index.css` and `src/components/import-dialog.css`.
