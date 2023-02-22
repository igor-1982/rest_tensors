program test
    implicit none

    real*8 :: eigenvector(10,10)
    real*8 :: ri3fn(10,10,20)
    real*8 :: ri3mo(20,10,10)

    eigenvector(:,:) = 1.0d0
    ri3fn(:,:,:) = 2.0d0
    ri3mo(:,:,:) = 0.0d0

    call ri_ao2mo_f(eigenvector,ri3fn,ri3mo,10,10,20)

    write(*,*) ri3mo
    
end program test


subroutine ri_ao2mo_f(eigenvector, ri3fn, ri3mo, num_states, num_basis, num_auxbas)

    implicit none

    integer :: num_states, num_basis, num_auxbas

    real*8 :: eigenvector(num_basis, num_states)
    real*8 :: ri3fn(num_basis, num_basis, num_auxbas)
    real*8 :: ri3mo(num_auxbas, num_states, num_states)

    !real*8, dimension(:,:), allocatable :: tmp_matr_1
    real*8, dimension(:,:), allocatable :: tmp_matr_2

    integer:: i_state, j_state, k_state

    !allocate(tmp_matr_1(num_basis, num_states))
    allocate(tmp_matr_2(num_basis, num_states))



    ri3mo(:,:,:) = 0.0d0

    do i_state = 1, num_auxbas, 1
        !tmp_matr_1(:,:) = ri3fn(:,:,i_state)
        tmp_matr_2(:,:) = 0.0d0
        call dgemm('N', 'N', num_basis, num_states, num_basis, 1.0d0, &
                ri3fn(:,:,i_state), num_basis, eigenvector(:,:), num_basis, &
                0.0d0, tmp_matr_2(:,:), num_basis)
        call dgemm('T','N', num_states, num_states, num_basis, 1.0d0, &
                eigenvector(:,:), num_basis, tmp_matr_2, num_basis, &
                0.0d0, ri3mo(i_state,:,:), num_basis)
    enddo

    !deallocate(tmp_matr_1, tmp_matr_2)
    deallocate(tmp_matr_2)

end subroutine ri_ao2mo_f


!subroutine copy_matr(f_matr, t_matr, fx_start, fx_end, fy_start, fy_end, tx_start, tx_end, ty_start, ty_end)
!    implicit none
!end subroutine copy_matr