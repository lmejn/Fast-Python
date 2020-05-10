#!/usr/bin/sh

f2py -m fortran_code -c fortran_src/interpolate.f90 fortran_src/streamtracer.f90
