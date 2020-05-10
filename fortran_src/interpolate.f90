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
end module interpolate