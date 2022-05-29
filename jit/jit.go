package jit

// #include "cgotorch/cgotorch.h"
import "C"
import (
	torch "github.com/wangkuiyi/gotorch"
	"log"
	"os"
	"runtime"
	"unsafe"
)

type Module struct {
	module C.JitModule
}

func Load(path string) Module {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		log.Fatal("no such a path ", path)
	}
	gomodule := Module{}
	gomodule.module = C.JitLoad(C.CString(path))
	return gomodule
}

func (module Module) Forward(input torch.Tensor) torch.Tensor {
	var t C.Tensor
	torch.MustNil(unsafe.Pointer(C.JitForward(C.Tensor(*input.T), &t)))
	runtime.KeepAlive(input.T)
	torch.SetTensorFinalizer((*unsafe.Pointer)(&t))
	return torch.Tensor{(*unsafe.Pointer)(&t)}
}
