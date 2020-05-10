module interpolate
    implicit none

    contains

    integer function locate_in_grid(xi, x_grid, n_grid)
        double precision, intent(in) :: xi
        double precision, intent(in), dimension(0:n_grid-1) :: x_grid
        integer, intent(in) :: n_grid
        integer :: i0, im, i1, i

        if(xi < x_grid(0)) then
            locate_in_grid = 0
            return
        end if
        if(x_grid(n_grid-2) <= xi) then
            locate_in_grid = n_grid-2
            return
        end if

        i0 = 0
        i1 = n_grid-1

        do i=1,n_grid
            im = floor(0.5*(i0 + i1))

            if( (x_grid(im).le.xi).and.(xi.lt.x_grid(im+1)) ) then
                locate_in_grid = im
                return
            end if
            
            if( xi.lt.x_grid(im) ) i1 = im
            if( xi.ge.x_grid(im+1) ) i0 = im+1
        end do

        locate_in_grid = 0
        return
    end function locate_in_grid

    double precision function trilinear_interp(xl, yl, zl, fc)
    double precision, intent(in) :: xl, yl, zl
    double precision, intent(in), dimension(0:1,0:1,0:1) :: fc
    double precision :: m_xl, m_yl, c00, c01, c10, c11, c0, c1

    m_xl = 1 - xl
    m_yl = 1 - yl

    c00 = fc(0, 0, 0)*m_xl + fc(1, 0, 0)*xl
    c01 = fc(0, 0, 1)*m_xl + fc(1, 0, 1)*xl
    c10 = fc(0, 1, 0)*m_xl + fc(1, 1, 0)*xl
    c11 = fc(0, 1, 1)*m_xl + fc(1, 1, 1)*xl

    c0 = c00*m_yl + c10*yl
    c1 = c01*m_yl + c11*yl

    trilinear_interp = c0*(1 - zl) + c1*zl
    return

    end function trilinear_interp

end module interpolate