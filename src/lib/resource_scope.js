// A distinct cache/pool key per inference instance, forwarding to the host's
// real GPUDevice. GPU resources themselves are never wrapped (WebIDL requires
// real GPUBuffer objects). Native methods/accessors receive their real owner.
const scopes = new WeakMap();

export function retainResource(device, resource) {
  scopes.get(device)?.retain(resource);
  return resource;
}

export function destroyResource(device, resource) {
  scopes.get(device)?.forget(resource);
  resource.destroy();
}

export function createResourceScope(device) {
  const persistent = new Set();
  const transient = new Set();
  let running = false;
  let disposed = false;
  const scopedDevice = new Proxy(device, {
    get(target, key) {
      const value = Reflect.get(target, key, target);
      if (key === 'destroy') return () => { throw new Error('MoGe does not own the GPUDevice'); };
      if (key === 'createBuffer' || key === 'createQuerySet') {
        return (...args) => {
          if (disposed) throw new Error('MoGe resource scope is disposed');
          const resource = value.apply(target, args);
          (running ? transient : persistent).add(resource);
          return resource;
        };
      }
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  const scope = {
    device: scopedDevice,
    retain(resource) { transient.delete(resource); persistent.add(resource); },
    forget(resource) { transient.delete(resource); persistent.delete(resource); },
    beginRun() { running = true; },
    endRun() {
      for (const resource of transient) resource.destroy();
      transient.clear();
      running = false;
    },
    dispose() {
      scope.endRun();
      for (const resource of persistent) resource.destroy();
      persistent.clear();
      disposed = true;
    },
  };
  scopes.set(scopedDevice, scope);
  return scope;
}
