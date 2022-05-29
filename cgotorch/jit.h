#include "cgotorch/torch.h"

#ifdef __cplusplus
extern "C" {
#endif
  typedef void* JitModule;

  JitModule JitLoad(const char *path);
  const char *JitSave(Tensor tensor, const char *path);
  const char *JitForward(JitModule module, Tensor input, Tensor *result);

#ifdef __cplusplus
}
#endif