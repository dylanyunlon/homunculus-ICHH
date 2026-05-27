//go:build amd64 && !purego

#include "textflag.h"

// func squaredL2AVX2(a, b []float32) float32
TEXT ·squaredL2AVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), AX
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), BX

	XORQ DX, DX
	VXORPS Y0, Y0, Y0

	MOVQ CX, SI
	SHRQ $3, SI
	JZ l2_reduce

l2_loop:
	VMOVUPS (AX)(DX*1), Y1
	VMOVUPS (BX)(DX*1), Y2
	VSUBPS Y2, Y1, Y1
	VMULPS Y1, Y1, Y1
	VADDPS Y1, Y0, Y0
	ADDQ $32, DX
	DECQ SI
	JNZ l2_loop

l2_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	ANDQ $7, CX
	JZ l2_done

l2_tail:
	MOVSS (AX)(DX*1), X1
	MOVSS (BX)(DX*1), X2
	SUBSS X2, X1
	MULSS X1, X1
	ADDSS X1, X0
	ADDQ $4, DX
	DECQ CX
	JNZ l2_tail

l2_done:
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET

// func dotAVX2(a, b []float32) float32
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), AX
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), BX

	XORQ DX, DX
	VXORPS Y0, Y0, Y0

	MOVQ CX, SI
	SHRQ $3, SI
	JZ dot_reduce

dot_loop:
	VMOVUPS (AX)(DX*1), Y1
	VMOVUPS (BX)(DX*1), Y2
	VMULPS Y2, Y1, Y1
	VADDPS Y1, Y0, Y0
	ADDQ $32, DX
	DECQ SI
	JNZ dot_loop

dot_reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS X1, X0, X0
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0

	ANDQ $7, CX
	JZ dot_done

dot_tail:
	MOVSS (AX)(DX*1), X1
	MOVSS (BX)(DX*1), X2
	MULSS X2, X1
	ADDSS X1, X0
	ADDQ $4, DX
	DECQ CX
	JNZ dot_tail

dot_done:
	VZEROUPPER
	MOVSS X0, ret+48(FP)
	RET
