/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

// Perform intersection queries using 2D triangles on a regular mesh as
// primitives and intersection with points as queries. One point per triangle.
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________
// |\x|\x|\x|
// |x\|x\|x\|
// __________

struct Triangle
{
  ArborX::Point a;
  ArborX::Point b;
  ArborX::Point c;
};

struct Mapping
{
  ArborX::Point alpha;
  ArborX::Point beta;
  ArborX::Point p0;

  KOKKOS_FUNCTION
  ArborX::Point get_coeff(ArborX::Point p) const
  {
    float alpha_coeff = alpha[0] * (p[0] - p0[0]) + alpha[1] * (p[1] - p0[1]) +
                        alpha[2] * (p[2] - p0[2]);
    float beta_coeff = beta[0] * (p[0] - p0[0]) + beta[1] * (p[1] - p0[1]) +
                       beta[2] * (p[2] - p0[2]);
    return {1 - alpha_coeff - beta_coeff, alpha_coeff, beta_coeff};
  }

  // x = a + alpha * (b - a) + beta * (c - a)
  //   = (1-beta-alpha) * a + alpha * b + beta * c
  //
  // FIXME Only works for 2D reliably
  void compute(const Triangle &triangle)
  {
    const auto &a = triangle.a;
    const auto &b = triangle.b;
    const auto &c = triangle.c;

    ArborX::Point u = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    ArborX::Point v = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};

    const float inv_det = 1. / (v[1] * u[0] - v[0] * u[1]);

    alpha = ArborX::Point{v[1] * inv_det, -v[0] * inv_det, 0};
    beta = ArborX::Point{-u[1] * inv_det, u[0] * inv_det, 0};
    p0 = a;
  }

  Triangle get_triangle() const
  {
    const float inv_det = 1. / (alpha[0] * beta[1] - alpha[1] * beta[0]);
    ArborX::Point a = p0;
    ArborX::Point b = {{p0[0] + inv_det * beta[1], p0[1] - inv_det * beta[0]}};
    ArborX::Point c = {
        {p0[0] - inv_det * alpha[1], p0[1] + inv_det * alpha[0]}};
    return {a, b, c};
  }
};

template <typename DeviceType>
class Points
{
public:
  Points(typename DeviceType::execution_space const &execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 2;
    int ny = 2;
    int n = nx * ny;
    float hx = Lx / (nx - 1);
    float hy = Ly / (ny - 1);

    auto index = [nx, ny](int i, int j) { return i + j * nx; };

    points_ = Kokkos::View<ArborX::Point *, typename DeviceType::memory_space>("points", 2 * n);
    auto points_host = Kokkos::create_mirror_view(points_);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
        points_host[2 * index(i, j)] = {(i + .252f) * hx, (j + .259f) * hy, 0.f};
        points_host[2 * index(i, j) + 1] = {(i + .75f) * hx, (j + .75f) * hy, 0.f};
        printf("(%.2f, %.2f), (%.2f, %.2f)\n", 
            points_host[2*index(i,j)][0], points_host[2*index(i,j)][1],
            points_host[2*index(i,j)+1][0], points_host[2*index(i,j)+1][1]
            );
      }
    Kokkos::deep_copy(execution_space, points_, points_host);
  }

  KOKKOS_FUNCTION auto const &get_point(int i) const { return points_(i); }

  KOKKOS_FUNCTION auto size() const { return points_.size(); }

private:
  Kokkos::View<ArborX::Point *, typename DeviceType::memory_space> points_;
};

template <typename DeviceType>
class Triangles
{
public:
  // Create non-intersecting triangles on a 3D cartesian grid
  // used both for queries and predicates.
  Triangles(typename DeviceType::execution_space const &execution_space)
  {
    float Lx = 100.0;
    float Ly = 100.0;
    int nx = 2;
    int ny = 2;
    int n = nx * ny;
    float hx = Lx / (nx);
    float hy = Ly / (ny);

    auto index = [nx, ny](int i, int j) { return i + j * nx; };

    triangles_ = Kokkos::View<Triangle *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "triangles"), 2 * n);
    auto triangles_host = Kokkos::create_mirror_view(triangles_);

    mappings_ = Kokkos::View<Mapping *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "mappings"), 2 * n);
    auto mappings_host = Kokkos::create_mirror_view(mappings_);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
      {
        ArborX::Point bl{i * hx, j * hy, 0.};
        ArborX::Point br{(i + 1) * hx, j * hy, 0.};
        ArborX::Point tl{i * hx, (j + 1) * hy, 0.};
        ArborX::Point tr{(i + 1) * hx, (j + 1) * hy, 0.};

        printf("1: [(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)], \
                \n2: [(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)]\n",
            tl[0], tl[1], tl[2],
            bl[0], bl[1], bl[2],
            br[0], br[1], br[2],
            tl[0], tl[1], tl[2],
            br[0], br[1], br[2],
            tr[0], tr[1], tr[2]);

        triangles_host[2 * index(i, j)] = {tl, bl, br};
        triangles_host[2 * index(i, j) + 1] = {tl, br, tr};
      }

    for (int k = 0; k < 2 * n; ++k)
    {
      mappings_host[k].compute(triangles_host[k]);

      Triangle recover_triangle = mappings_host[k].get_triangle();

      for (unsigned int i = 0; i < 3; ++i)
        if (std::abs(triangles_host[k].a[i] - recover_triangle.a[i]) > 1.e-3)
          abort();

      for (unsigned int i = 0; i < 3; ++i)
        if (std::abs(triangles_host[k].b[i] - recover_triangle.b[i]) > 1.e-3)
          abort();

      for (unsigned int i = 0; i < 3; ++i)
        if (std::abs(triangles_host[k].c[i] - recover_triangle.c[i]) > 1.e-3)
          abort();
    }
    Kokkos::deep_copy(execution_space, triangles_, triangles_host);
    Kokkos::deep_copy(execution_space, mappings_, mappings_host);
  }

  // Return the number of triangles.
  KOKKOS_FUNCTION int size() const { return triangles_.size(); }

  // Return the triangle with index i.
  KOKKOS_FUNCTION const Triangle &get_triangle(int i) const
  {
    return triangles_(i);
  }

  KOKKOS_FUNCTION const Mapping &get_mapping(int i) const
  {
    return mappings_(i);
  }

private:
  Kokkos::View<Triangle *, typename DeviceType::memory_space> triangles_;
  Kokkos::View<Mapping *, typename DeviceType::memory_space> mappings_;
};

// For creating the bounding volume hierarchy given a Triangles object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Triangles class, we just resort to them.
template <typename DeviceType>
struct ArborX::AccessTraits<Triangles<DeviceType>, ArborX::PrimitivesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Triangles<DeviceType> const &triangles)
  {
    return triangles.size();
  }
  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
  {
    const auto &triangle = triangles.get_triangle(i);
    ArborX::Box box{};
    box += triangle.a;
    box += triangle.b;
    box += triangle.c;
    return box;
  }
};

//// For performing the queries given a Triangles object, we need to define memory
//// space, how to get the total number of queries, and what the query with index
//// i should look like. Since we are using self-intersection (which boxes
//// intersect with the given one), the functions here very much look like the
//// ones in ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag>.
//template <typename DeviceType>
//struct ArborX::AccessTraits<Triangles<DeviceType>, ArborX::PredicatesTag>
//{
//  using memory_space = typename DeviceType::memory_space;
//  static KOKKOS_FUNCTION int size(Triangles<DeviceType> const &triangles)
//  {
//    return triangles.size();
//  }
//  static KOKKOS_FUNCTION auto get(Triangles<DeviceType> const &triangles, int i)
//  {
//    const auto &triangle = triangles.get_triangle(i);
//    ArborX::Box box{};
//    box += triangle.a;
//    box += triangle.b;
//    box += triangle.c;
//    //return intersects(box);
//    return intersects(box);
//  }
//};
template <typename DeviceType>
struct ArborX::AccessTraits<Points<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Points<DeviceType> const &points)
  {
    return points.size();
  }
  static KOKKOS_FUNCTION auto get(Points<DeviceType> const &points, int i)
  {
    const auto& point = points.get_point(i);
    return intersects(point);
  }
};


template <typename DeviceType>
class TriangleIntersectionCallback
{
public:
  TriangleIntersectionCallback(Triangles<DeviceType> triangles)
      : triangles_(triangles)
  {
  }

  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int primitive_index,
                                  OutputFunctor const &out) const
  {

    const ArborX::Point &point = getGeometry(predicate);
    //auto predicate_index = ArborX::getData(predicate);
    const auto coeffs = triangles_.get_mapping(primitive_index).get_coeff(point);
    bool intersects = coeffs[0] >= 0 && coeffs[1] >= 0 && coeffs[2] >= 0;
    auto triangle = triangles_.get_triangle(primitive_index);
    printf("(%f, %f), \
            [(%f, %f, %f), (%f, %f, %f), (%f, %f, %f)], %f, %f, %f\n",
            point[0], point[1],
            triangle.a[0], triangle.a[1], triangle.a[2],
            triangle.b[0], triangle.b[1], triangle.b[2],
            triangle.c[0], triangle.c[1], triangle.c[2],
            coeffs[0], coeffs[1], coeffs[2]
            );
    if(intersects) {
      out(primitive_index);
    }
    else {
      printf("not intersects\n");
    }
  }

private:
  Triangles<DeviceType> triangles_;
};

// Now that we have encapsulated the objects and queries to be used within the
// Triangles class, we can continue with performing the actual search.
int main()
{
  Kokkos::initialize();
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with triangles.\n";
    Triangles<DeviceType> triangles(execution_space);

    constexpr float eps = 1.e-3;

    //for (int i = 0; i < triangles.size(); ++i)
    //{
    //  const auto &mapping = triangles.get_mapping(i);
    //  const auto &triangle = triangles.get_triangle(i);
    //  const auto &coeff_a = mapping.get_coeff(triangle.a);
    //  if ((std::abs(coeff_a[0] - 1.) > eps) || std::abs(coeff_a[1]) > eps ||
    //      std::abs(coeff_a[2]) > eps)
    //    std::cout << i << " a: " << coeff_a[0] << ' ' << coeff_a[1] << ' '
    //              << coeff_a[2] << std::endl;
    //  const auto &coeff_b = mapping.get_coeff(triangle.b);
    //  if ((std::abs(coeff_b[0]) > eps) || std::abs(coeff_b[1] - 1.) > eps ||
    //      std::abs(coeff_b[2]) > eps)
    //    std::cout << i << " b: " << coeff_b[0] << ' ' << coeff_b[1] << ' '
    //              << coeff_b[2] << std::endl;
    //  const auto &coeff_c = mapping.get_coeff(triangle.c);
    //  if ((std::abs(coeff_c[0]) > eps) || std::abs(coeff_c[1]) > eps ||
    //      std::abs(coeff_c[2] - 1.) > eps)
    //    std::cout << i << " c: " << coeff_c[0] << ' ' << coeff_c[1] << ' '
    //              << coeff_c[2] << std::endl;
    //}
    std::cout << "Triangles set up.\n";

    std::cout << "Creating BVH tree.\n";
    ArborX::BVH<MemorySpace> const tree(execution_space, triangles);
    std::cout << "BVH tree set up.\n";

    std::cout << "Create the points used for queries.\n";
    Points<DeviceType> points(execution_space);
    std::cout << "Points for queries set up.\n";

    std::cout << "Starting the queries.\n";
    int const n = points.size();
    printf("point size %d, triangle size %d\n", points.size(), triangles.size());
    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    //Kokkos::View<ArborX::Point *, MemorySpace> coefficients("coefficients", n);

    ArborX::query(tree, execution_space, points, TriangleIntersectionCallback{triangles}, indices, offsets);
    //ArborX::query(tree, execution_space, points, indices, offsets);
    //ArborX::query(tree, execution_space, points, indices, offsets);
    std::cout << "Queries done.\n";
    auto indices_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
    auto offsets_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);

    printf("indices %d\n", indices_h.extent(0));
    for(int i=0; i<indices_h.extent(0); ++i) {
      printf("%d ", indices_h(i));
    }
    printf("\noffsets %d\n", offsets_h.extent(0));
    for(int i=0; i<offsets_h.extent(0); ++i) {
      printf("%d ", offsets_h(i));
    }
    printf("\n");

    /*
    std::cout << "Starting checking results.\n";
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto coeffs_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, coefficients);

    for (int i = 0; i < n; ++i)
    {
      if (offsets_host(i) != i)
      {
        std::cout << offsets_host(i) << " should be " << i << std::endl;
      }
      const auto &c = coeffs_host(i);
      const auto &t = triangles_host.get_triangle(offsets_host(i));
      const auto &p_h = points.get_point(i);
      ArborX::Point p = {{c[0] * t.a[0] + c[1] * t.b[0] + c[2] * t.c[0]},
                         {c[0] * t.a[1] + c[1] * t.b[1] + c[2] * t.c[1]},
                         {c[0] * t.a[2] + c[1] * t.b[2] + c[2] * t.c[2]}};
      if ((std::abs(p[0] - p_h[0]) > eps) || std::abs(p[1] - p_h[1]) > eps ||
          std::abs(p[2] - p_h[2]) > eps)
      {
        std::cout << "coeffs for point " << i << " are wrong!\n";
      }
    }

    std::cout << "Checking results successful.\n";
    */
  }

  Kokkos::finalize();


}
