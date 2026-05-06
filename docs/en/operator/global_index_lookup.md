# Global Index Lookup Based on `decode_list`

## Goal

Given a global token index `global_index` for the current round, we want to map it back to:

* Which sequence this token belongs to
* Its sequence position `sequence_index` within that sequence

The data structure available today is `DecodeList`. You can think of it as a table of token ranges for the current round.

Each `SequenceSlice` in this table describes a continuous range:

* Where the range starts in global token space
* Which batch slot it belongs to
* Which starting position in the sequence it corresponds to
* How long the range is

`last_token_flag` is only a marker used by later output logic and does not participate in the reverse lookup.

---

## Core Idea

The problem is really just:

```text
Given global_index, find the slice that covers it.
```

Once the matching slice is found, the rest is just offset arithmetic.

If a slice says:

* The range starts at global position `token_start_index`
* It starts at sequence position `sequence_index`

Then for any global position inside that range:

$$
offset = global\_index - token\_start\_index
$$

So the position within the sequence is:

$$
sequence\_index + offset
$$

So the reverse lookup has two steps:

1. Find which slice contains `global_index`
2. Use the relative offset to recover the real sequence position

---

## How to Think About It

You can think of `decode_list` as a directory table created after concatenating multiple sequences in the current computation view.

Simple sketch:

```text
Global positions: 0 1 2 3 4 5 6 7
Owned by slice:   A A A A A A B B
```

This is equivalent to two ranges:

```text
slice A: [0, 6)
slice B: [6, 8)
```

If:

* slice A belongs to batch 0 and starts at sequence position 0
* slice B belongs to batch 1 and starts at sequence position 0

Then:

* Global position `4` falls in A, so it maps to batch 0, sequence position 4
* Global position `7` falls in B, so it maps to batch 1, sequence position 1

The key question is not "which token number is it," but "which range does it fall into."

---

## Why `decode_list` Can Do This

`decode_list` works for reverse lookup because it is already ordered by global token position.

That means:

* The slice order matches the global token order
* Each slice covers a continuous range
* The next slice always starts after the previous one

So it is naturally suitable for range lookup.

There is no need to build a huge array that explicitly maps every `global_index` back to a sequence.

---

## Single-Point Query

Single-point lookup is suitable when:

* A random `global_index` is given
* It is queried once, or only a few times

The most natural approach is:

* Find the first range whose right boundary is greater than `global_index`
* Then confirm that `global_index` is not to the left of that range

If it matches, you can recover:

* Which `batch_index` it belongs to
* The true position within the sequence
* Which slice it hit

If it does not match, then:

* `global_index` is out of range
* Or there is a hole between ranges and this position falls into that hole

So the essence of single-point lookup is:

```text
Locate the range first, then compute the offset inside that range.
```

---

## Continuous Range Query

In multithreaded execution, a more common case is not "query one random point" but "one thread owns a continuous global range."

For example, a thread may own:

$$
[global\_begin,\ global\_end)
$$

If every position in that range were queried from scratch, there would be a lot of repeated work.

A better approach is:

1. Locate the start position `global_begin` once
2. Get the current `slice_index`
3. Walk forward inside the current slice
4. After reaching the end of that slice, move to the next one

Simple sketch:

```text
Thread owns range: [4, 8)

Global positions: 0 1 2 3 4 5 6 7
Owned by slice:   A A A A A A B B
Thread visits:             ^ ^ ^ ^
```

This thread only needs to:

* Find that position `4` is in A
* Then advance to `5`
* Then switch to B and visit `6`, `7`

So the core of continuous range lookup is not repeated positioning, but:

```text
Locate once, then walk forward.
```

---

## Why Decode Is Simpler

In the decode round, each sequence contributes only one token in that round.

So each slice has length `1`, and the table almost collapses into:

```text
global position 0 -> slice 0
global position 1 -> slice 1
global position 2 -> slice 2
...
```

In other words, the range table still exists in decode mode, but each range contains only one point, so lookup is very direct.

---

## Why Prefill Needs Range Thinking

In the prefill round, a sequence may enter multiple consecutive tokens in one pass.

So a slice can be longer than `1`.

Then `decode_list` looks more like:

```text
slice A covers [0, 6)
slice B covers [6, 8)
slice C covers [8, 12)
```

At that point, global position and sequence position are no longer one-to-one and must be determined by "which range it falls into."

That is exactly what `lookup_global_index` and `walk_global_range` are meant to solve.

---

## Meaning of `slice_index`

In the lookup result, `slice_index` is kept in addition to `batch_index` and `sequence_index`.

Its job is not to represent the token position within the sequence, but to indicate:

```text
which slice in decode_list was matched
```

This is useful for walking a continuous range, because the next lookup does not have to start from the beginning again; it can continue from the current slice.

If the question is "which sequence was scheduled first this round," then under the current implementation `slice_index` can also be understood as the order number.

---

## Why the Complexity Is Reasonable

For a single-point query:

* You need one lookup in the range table
* The cost is $O(\log N)$

For a continuous range query:

* Only the start position is located once
* The rest is mostly sequential walking inside ranges

So the cost can be understood as:

* Start lookup: $O(\log N)$
* Range walk: $O(K + S)$

Where:

* $N$ is the length of `decode_list`
* $K$ is the number of global positions the thread actually handles
* $S$ is the number of slices crossed by that range

This fits the current thread model, where work is assigned in continuous ranges, much better than re-querying every single position.

---

## Final Recommendation

The most useful mental model is:

```text
Treat DecodeList as a global token range table.
For single-point queries, find the matching range first and then compute the offset.
For continuous range queries, locate the start once and walk forward along slices.
```

This matches both the current `BatchScheduler` data layout and the way multithreaded token processing is done in continuous ranges.
