// Project Dependencies
#include <jni.h>
#include <jnipp.h>

// Standard Dependencies
#include <cmath>

// Local Dependencies
#include "testing.h"

/*
    jni::Vm Tests
 */
TEST(Vm_externalCreateAndAttach) {
    JNIEnv *env;
    JavaVMInitArgs args = {};
    args.version = JNI_VERSION_1_2;
    JavaVM *javaVm{};
    auto ret = JNI_CreateJavaVM(&javaVm, (void **)&env, &args);
    ASSERT(ret == 0);

    {
        jni::init(env);
        jni::Class cls("java/lang/String");
    }
    JavaVM *localVmPointer{};

    ret = env->GetJavaVM(&localVmPointer);
    ASSERT(ret == 0);
}

int main() {
    // jni::Vm Tests
    RUN_TEST(Vm_externalCreateAndAttach);

    return 0;
}
