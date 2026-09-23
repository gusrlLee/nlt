import mitsuba as mi
import drjit as dr
import matplotlib.pyplot as plt

mi.set_variant('cuda_ad_rgb')

class PathTracer(mi.SamplingIntegrator):
    def __init__(self, props):
        super().__init__(props)
        self.max_depth = 5

    @dr.syntax
    def sample(self, scene, sampler, ray, medium=None, active=True):
        # init 
        bsdf_ctx = mi.BSDFContext()
        beta = mi.Spectrum(1.0)
        L = mi.Spectrum(0.0)
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        r = mi.Ray3f(ray)

        while (active):
            # bounce 
            si: mi.SurfaceInteraction3f = scene.ray_intersect(r, active)
            is_hit = si.is_valid()
            valid_ray = valid_ray | is_hit

            active = active & is_hit

            light = si.emitter(scene, active=active)
            is_light = light != None 

            Le = light.eval(si, active=(active & is_light))
            L = L + beta * Le

            bsdf = si.bsdf()
            bs, weight = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active)

            new_dir = si.to_world(bs.wo)
            next_ray = si.spawn_ray(new_dir)

            beta = beta * weight
            r = next_ray

            depth = depth + 1
            is_continue = depth < self.max_depth
            
            active = active & is_continue

        return L, valid_ray, []


if __name__ == "__main__":
    scene = mi.load_file("assets/cornell-box/scene.xml")
    integrator = PathTracer(mi.Properties())
    image = mi.render(scene, integrator=integrator, spp=16)

    plt.axis("off")
    plt.imshow(image ** (1.0 / 2.2))
    plt.show()