module streamtracer
    use interpolate
    implicit none

    contains

    subroutine update(xs, ix, x, y, z, vx, vy, vz, nx, ny, nz, ds, k)
        double precision, intent(in), dimension(0:2) :: xs
        integer, intent(inout), dimension(0:2) :: ix
        double precision, intent(in) :: x(0:nx-1), y(0:ny-1), z(0:nz-1)
        double precision, intent(in), dimension(0:nx-1,0:ny-1,0:nz-1) :: vx, vy, vz
        integer, intent(in) :: nx, ny, nz
        double precision, intent(in) :: ds
        double precision, intent(out), dimension(0:2) :: k
        double precision, dimension(0:1, 0:1, 0:1) :: vxc, vyc, vzc
        double precision :: xl, yl, zl, vxi, vyi, vzi, vmag

        ix(0) = locate_in_grid(xs(0), x, nx)
        ix(1) = locate_in_grid(xs(1), y, ny)
        ix(2) = locate_in_grid(xs(2), z, nz)
    
        vxc = vx(ix(0):ix(0)+1, ix(1):ix(1)+1, ix(2):ix(2)+1)
        vyc = vy(ix(0):ix(0)+1, ix(1):ix(1)+1, ix(2):ix(2)+1)
        vzc = vz(ix(0):ix(0)+1, ix(1):ix(1)+1, ix(2):ix(2)+1)
    
        xl = scale_to_grid(xs(0), x(ix(0)), x(ix(0)+1))
        yl = scale_to_grid(xs(1), x(ix(1)), x(ix(1)+1))
        zl = scale_to_grid(xs(2), x(ix(2)), x(ix(2)+1))
    
        vxi = trilinear_interp(xl, yl, zl, vxc)
        vyi = trilinear_interp(xl, yl, zl, vyc)
        vzi = trilinear_interp(xl, yl, zl, vzc)
    
        vmag = sqrt(vxi**2 + vyi**2 + vzi**2)

        if(vmag.eq.0.) then
            k = 0.
        else
            k(0) = ds*vxi/vmag
            k(1) = ds*vyi/vmag
            k(2) = ds*vzi/vmag
        end if
        
        return
    end subroutine update

    subroutine calc_streamline(xs0, &
                               x, y, z, vx, vy, vz, nx, ny, nz, &
                               s, ns, step_size, xs, n_steps)
    double precision, intent(in), dimension(0:2) :: xs0
    double precision, intent(in) :: x(0:nx-1), y(0:ny-1), z(0:nz-1)
    double precision :: s(0:ns-1)
    double precision, intent(in), dimension(0:nx-1,0:ny-1,0:nz-1) :: vx, vy, vz
    integer, intent(in) :: nx, ny, nz, ns
    double precision, intent(in) :: step_size
    double precision, intent(out), dimension(0:ns-1, 0:2) :: xs
    integer, intent(out) :: n_steps
    double precision, dimension(0:2) :: xi, k1, k2, k3, k4
    double precision :: dx, dy, dz, ds
    integer, dimension(0:2) :: ix
    integer :: i

    xi = xs0
    ix = 0

    do i=0,ns-1

        ! Check bounds
        if((xi(0).lt.x(0)).or.(x(nx-1).le.xi(0)).or. &
           (xi(1).lt.y(0)).or.(y(ny-1).le.xi(1)).or. &
           (xi(2).lt.z(0)).or.(z(nz-1).le.xi(2))) then
           exit
        end if
        
        ! Calculate step size
        ix(0) = locate_in_grid(xi(0), x, nx)
        ix(1) = locate_in_grid(xi(1), y, ny)
        ix(2) = locate_in_grid(xi(2), z, nz)

        dx = x(ix(0)+1) - x(ix(0))
        dy = y(ix(1)+1) - y(ix(1))
        dz = z(ix(2)+1) - z(ix(2))

        ds = step_size*min(dx, dy, dz)

        ! RK4

        call update(xi,          ix, x, y, z, vx, vy, vz, nx, ny, nz, ds, k1)
        call update(xi + 0.5*k1, ix, x, y, z, vx, vy, vz, nx, ny, nz, ds, k2)
        call update(xi + 0.5*k2, ix, x, y, z, vx, vy, vz, nx, ny, nz, ds, k3)
        call update(xi + k3,     ix, x, y, z, vx, vy, vz, nx, ny, nz, ds, k4)

        xi = xi + (k1 + 2*(k2 + k3) + k4)/6.

        ! Save to arrays

        xs(i, :) = xi
        s(i) = ds
    end do
    
    n_steps = i
    
                            
    end subroutine calc_streamline
end module streamtracer