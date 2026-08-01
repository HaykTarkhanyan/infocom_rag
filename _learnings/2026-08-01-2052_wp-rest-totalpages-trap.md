# WordPress `X-WP-TotalPages` reflects the request's own `per_page`

**Symptom.** The fetcher probed the collection size, then walked pages and died
partway through:

```
Categories 1085,1083 contain 94 posts (94 pages)
  page 1/94: 94 posts
  HTTP 400 for /posts (attempt 1/3)
  ...
Fetch failed: Failed to GET /posts after 3 attempts
```

**Cause.** The probe used `per_page=1` to be cheap, so the API returned
`X-WP-Total: 94` and `X-WP-TotalPages: 94` -- 94 pages *of one post each*. The
actual fetch then used `per_page=100`, which needs only 1 page. Requesting page 2
is past the end, and WordPress answers `400 rest_post_invalid_page_number` rather
than an empty list.

**Fix.** Never carry `X-WP-TotalPages` across a `per_page` change. Derive it:

```python
total = int(probe.headers["X-WP-Total"])
total_pages = math.ceil(total / PER_PAGE)
```

**Also worth knowing.** Category counts do not sum to the corpus size, because a
post carries its subcategory *and* its parent. Querying `?categories=51` returns
the whole `indepth` subtree. Our two categories list 63 + 41 = 104 posts but the
union is **94** -- 10 articles belong to both. Deduplicate by post id and treat
per-category counts as overlapping sets.
