// src/__tests__/server.test.js
import request from 'supertest';
import fs from 'fs';
import path from 'path';
import { app } from '../server.js';    // ← one “..” instead of two

// point this at wherever your KB lives in production
const kbFile = path.join(process.cwd(), 'public', 'knowledge_base.json');

beforeEach(() => {
  // reset KB to a known empty state before each test
  fs.writeFileSync(
    kbFile,
    JSON.stringify({ rawData: [], embeddings: [], centroids: {} }, null, 2),
    'utf-8'
  );
});

describe('GET /', () => {
  it('serves index.html', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
    expect(res.text).toContain('<!DOCTYPE html>');
  });
});

describe('POST /send', () => {
  it('returns 400 when no data field is provided', async () => {
    const res = await request(app)
      .post('/send')
      .send({})
      .set('Content-Type', 'application/json');
    expect(res.statusCode).toBe(400);
    expect(res.body).toEqual({ error: 'Missing "data" in request body.' });
  });

  it('creates a single zero centroid for one token', async () => {
    const res = await request(app)
      .post('/send')
      .send({ data: 'foobar' })
      .set('Content-Type', 'application/json');
    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('centroids');
    const centroids = res.body.centroids;
    // with one token, your x-means fallback gives exactly one centroid “1”
    expect(Object.keys(centroids)).toEqual(['1']);
    expect(centroids['1']).toHaveLength(100);
  });

  it('clusters two tokens into at least one centroid', async () => {
    const res = await request(app)
      .post('/send')
      .send({ data: 'hello world' })
      .set('Content-Type', 'application/json');
    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('centroids');
    const keys = Object.keys(res.body.centroids);
    expect(keys.length).toBeGreaterThanOrEqual(1);
  });
});
