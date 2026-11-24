"""Lee un state_dict de PyTorch (.pt en formato zip) sin tener PyTorch instalado."""
import pickle, zipfile, sys
from collections import OrderedDict
import numpy as np

TIPOS = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64,
         "IntStorage": np.int32, "HalfStorage": np.float16, "BoolStorage": np.bool_}

def cargar_state_dict(ruta):
    z = zipfile.ZipFile(ruta)
    raiz = z.namelist()[0].split("/")[0]
    def rebuild(storage, offset, size, stride, *args):
        arr = storage
        if len(size) == 0:
            return np.array(arr[offset])
        return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                               strides=tuple(s * arr.itemsize for s in stride)).copy()
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                return OrderedDict
            if mod == "torch" and name in TIPOS:
                return name
            if mod == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, hooks: data
            raise pickle.UnpicklingError(f"clase no permitida: {mod}.{name}")
        def persistent_load(self, pid):
            _, tipo, clave, _loc, numel = pid
            dtype = TIPOS[tipo if isinstance(tipo, str) else tipo.__name__]
            datos = z.read(f"{raiz}/data/{clave}")
            return np.frombuffer(datos, dtype=dtype)[:numel]
    return U(z.open(f"{raiz}/data.pkl")).load()

if __name__ == "__main__":
    sd = cargar_state_dict(sys.argv[1])
    for k, v in sd.items():
        print(k, getattr(v, "shape", v), getattr(v, "dtype", ""))
