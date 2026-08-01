# selectolax text extraction needs `separator="", strip=False`

When pulling text out of WordPress `content.rendered` with selectolax, the
`.text()` arguments are not cosmetic -- the two obvious choices are both wrong,
and either corrupts tokenization on every chunk downstream.

**`strip=True`** strips whitespace from each text node before joining, which
glues words across `<span>` boundaries:

```
բախում։Ըստ          # wrong -- the space after ։ lived in the span's tail
```

**`separator=" "`** inserts a space at every inline boundary, which detaches
Armenian punctuation from the word it belongs to:

```
օրենքի ՝            # wrong -- should be օրենքի՝
```

**`separator="", strip=False`** preserves the HTML's own whitespace, which is
authoritative, then collapse runs afterwards:

```python
text = re.sub(r"\s+", " ", node.text(separator="", strip=False)).strip()
```

Verified on the paragraph that exposed both bugs:

```
"բախում։ Ըստ" present : True
"օրենքի՝" present     : True
"օրենքի ՝" absent     : True
```

**Why it matters.** `օրենքի ՝` and `օրենքի՝` tokenize differently, so the defect
propagates into every embedding, silently degrading retrieval. It is not a
display issue.

Related: the source HTML is Elementor page-builder markup, so a regex tag-strip
is not sufficient either -- `li.elementor-icon-list-item` is the post-info widget
and leaks time, date and byline into the top of every article unless removed by
class before extraction.
