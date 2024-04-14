import os

import torch
import triton
import triton.language as tl

DEBUG = True

if DEBUG:
    os.environ["TRITON_INTERPRET"] = "1"


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit()  # set interpret=DEBUG when it's already here
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # Note that TMP is a scratchpad buffer to workaround a compiler bug, (I think TMP means temporary?)
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,  # m because probably it's a MxN matrix, Hence the Query needs to be M size and the Key needs to be N size
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,  # Just think of output and query is the same thing. So it's also operates in the M size
    nheads,
    seqlen_q,
    seqlen_k,  # probably value is also has the same seqlen size as key
    seqlen_q_rounded,  # don't know why we need this. Probably for padding for query later? Or it's because the design of the GPU?
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    # Everything that's m is row and everythings that's n is column
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (
            offs_m[:, None] * stride_qm
            + offs_d[
                None, :
            ]  # I don't understand why we need to add offs_d on the last operation
        )
    )
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + (
            offs_m[:, None] * stride_kn
            + offs_d[
                None, :
            ]  # I don't understand why we need to add offs_d on the last operation
        )
    )
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_h * stride_vh
        + (
            offs_m[:, None] * stride_vn
            + offs_d[
                None, :
            ]  # I don't understand why we need to add offs_d on the last operation
        )
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize poniter to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # current max
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in the SRAM throughout
    # [2022-10-30] TD: bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            # We need mask in here because it's not even headdim. So we need to mask it
            q = tl.load(
                q_ptrs,
                mask=offs_d[None, :] < headdim,
                other=0.0,
            )
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=(offs_m[None, :] < seqlen_q), other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[None, :] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_M)
        # -- compute qk ----
        if (
            EVEN_N & EVEN_M
        ):  # If we just do "if EVEN_N", there's seems to be some race conditions
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(offs_d[None, :] < headdim),
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(offs_n[None, :] < seqlen_k),
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(offs_n[None, :] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:
            qk += tl.where((start_n + offs_n[None, :]) < seqlen_k, 0.0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(
                (offs_m[:, None] >= (start_n + offs_n)[None, :]), 0.0, float("-inf")
            )
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(offs_n < seqlen_k), other=0.0
                    ).to(tl.float32)
                # Convert to matrix
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((offs_n + start_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma (fused multiply-add) instruction. But if we have bias
            # we need to multiply with the softmax scale here

            # In here, they implemented safe softmax
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)  # new max
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])  # dots_exp
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # Bug : Have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]  # output
        # update acc_o
        if (
            EVEN_N & EVEN_M
        ):  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(offs_d[None, :] < headdim),
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k)
                    & (offs_d[None, :] < headdim),
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)  # o

        # update statistics
        m_i = m_ij  # cur_max = new_max
        l_i_new = tl.exp(lse_i - m_ij) + l_ij  #
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG : have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    # Rematerialize offsets to save registers
    start_m = tl.program_id(0)
