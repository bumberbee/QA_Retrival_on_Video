-- Lecture RAG on Supabase (Postgres 15+ with pgvector). Run once in the SQL editor, or:
--   psql "$SUPABASE_DB_URL" -f deploy/supabase/schema.sql
-- Re-running is safe: every object is created only if it does not exist yet.
-- Files (video mp4, keyframe jpg) live in Cloudflare R2; the *_key columns hold their object keys.

create extension if not exists vector;
create extension if not exists unaccent;

-- unaccent() is not immutable, so it cannot back a generated column or an index directly.
create or replace function public.f_unaccent(text) returns text
language sql immutable parallel safe strict
as $$ select public.unaccent('public.unaccent'::regdictionary, $1) $$;

-- One lecture video.
create table if not exists videos (
  id              text primary key,                 -- same 12-hex id as the file-based app
  filename        text not null,                    -- original file name, e.g. L25_V011.mp4
  language        text,                             -- 'vi' / 'en'
  duration_sec    real,
  status          text not null default 'ready'     -- ready / processing / failed
                  check (status in ('ready', 'processing', 'failed')),
  video_key       text,                             -- R2 key of the mp4
  source_url      text,
  source_platform text check (source_platform in ('youtube', 'facebook', 'tiktok')),
  has_transcript  boolean not null default false,
  has_slides      boolean not null default false,
  meta            jsonb not null default '{}',      -- slide variants, pipeline manifests, ...
  uploaded_at     timestamptz not null default now()
);

-- One slide (stage-04 segment): its time span, cleaned OCR text and the speech said over it.
create table if not exists segments (
  id           text primary key,                    -- segment_id, e.g. L25_V011:seg_00005
  video_id     text not null references videos(id) on delete cascade,
  seq          integer not null,
  start_sec    real not null,
  end_sec      real not null,
  kind         text not null default 'slide',       -- slide / speaker / other / ad ...
  ad_flagged   boolean not null default false,
  title        text not null default '',
  ocr_text     text not null default '',
  speech_text  text not null default '',
  image_key    text,                                -- R2 key of the keyframe
  extra        jsonb not null default '{}'          -- regions, timings, anything else of stage 04
);
create index if not exists segments_video_idx on segments (video_id, seq);

-- Transcript shown in the "Bản chép lời" tab.
create table if not exists transcript_segments (
  video_id   text not null references videos(id) on delete cascade,
  idx        integer not null,
  start_sec  real not null,
  end_sec    real not null,
  text       text not null,
  primary key (video_id, idx)
);

-- Search evidence (stage-05 chunks): E5 vector + full-text vector without diacritics.
create table if not exists chunks (
  id           text primary key,                    -- chunk_id
  video_id     text not null references videos(id) on delete cascade,
  segment_id   text not null references segments(id) on delete cascade,
  source_type  text not null check (source_type in ('speech', 'ocr', 'region')),
  start_sec    real not null,
  end_sec      real not null,
  title        text not null default '',
  text         text not null,
  embedding    vector(384),                         -- multilingual-e5-small, L2-normalized
  tsv          tsvector generated always as
               (to_tsvector('simple', public.f_unaccent(coalesce(title, '') || ' ' || text))) stored
);
create index if not exists chunks_video_idx on chunks (video_id);
create index if not exists chunks_segment_idx on chunks (segment_id);
create index if not exists chunks_tsv_idx on chunks using gin (tsv);
create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);

-- Slide-review labels of the web app (variant x slide).
create table if not exists slide_reviews (
  video_id   text not null references videos(id) on delete cascade,
  variant    text not null,
  slide_key  text not null,
  verdict    text,
  note       text not null default '',
  updated_at timestamptz not null default now(),
  primary key (video_id, variant, slide_key)
);

-- Extraction queue for "upload = extract" (deploy step 5).
create table if not exists jobs (
  id          bigserial primary key,
  video_id    text not null references videos(id) on delete cascade,
  status      text not null default 'queued'
              check (status in ('queued', 'running', 'done', 'failed')),
  step        text,                                 -- 01 slides / 02 ocr / ...
  progress    real,                                 -- 0..1
  error       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);
create index if not exists jobs_status_idx on jobs (status, created_at);

-- Small settings the browser may read, e.g. the current backend URL of the Vast machine.
create table if not exists app_config (
  key        text primary key,
  value      text not null,
  updated_at timestamptz not null default now()
);

-- Row level security: the backend connects with the database URL (bypasses RLS); the browser,
-- with the public anon key, may only read app_config.
alter table videos              enable row level security;
alter table segments            enable row level security;
alter table transcript_segments enable row level security;
alter table chunks              enable row level security;
alter table slide_reviews       enable row level security;
alter table jobs                enable row level security;
alter table app_config          enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where tablename = 'app_config' and policyname = 'app_config read') then
    create policy "app_config read" on app_config for select using (true);
  end if;
end $$;
