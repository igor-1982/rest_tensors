program test
    implicit none

    !real*8 :: eigenvector(10,10)
    !real*8 :: ri3fn(10,10,20)
    !real*8 :: ri3mo(20,10,10)

    !eigenvector(:,:) = 1.0d0
    !ri3fn(:,:,:) = 2.0d0
    !ri3mo(:,:,:) = 0.0d0

    !call ri_ao2mo_f(eigenvector,ri3fn,ri3mo,10,10,20)

    !write(*,*) ri3mo

    real*8 :: matr_a(3,3), matr_b(3,3), matr_c(3,3)
    real*8 :: smatr_a(2,2), smatr_b(2,2), smatr_c(2,2)
    integer :: i,j

    matr_c(:,:) = 2.0d0
    do i = 1, 4
        do j = 1,4
            matr_a(i,j) = i + 3.0*(j-1.0)
            matr_b(i,j) = matr_a(i,j) + 5.0
        end do
    end do
    !write(*,*) matr_a, matr_b, matr_c
    write(*,*) matr_a(2,2), matr_a(2,3)
    write(*,*) matr_a(3,2), matr_a(3,3)
    write(*,*) matr_b(1,1), matr_b(1,2)
    write(*,*) matr_b(2,1), matr_b(2,2)

    call dgemm('N', 'N', 2,2,2, &
        1.0d0, &
        matr_a(2:3,2:3), 2, &
        matr_b(1:2,1:2), 2, &
        1.0d0,  &
        matr_c(2:3,1:2), 2 &
        )
    !call general_dgemm_f(&
    !    matr_a, 3, 3, 1, 2, 1, 2, 'N', &
    !    matr_b, 3, 3, 0, 2, 0, 2, 'N', &
    !    matr_c, 3, 3, 1, 2, 0, 2, &
    !    1.0d0, 1.0d0 &
    !)

    write(*,*) matr_c(2,1), matr_c(2,2)
    write(*,*) matr_c(3,1), matr_c(3,2)

    !smatr_a(:,:) = matr_a(2:3,2:3)
    !smatr_b(:,:) = matr_b(1:2,1:2)
    !smatr_c(:,:) = 2.0d0
    !call dgemm('N', 'N', 3,3,3, &
    !    1.0d0, &
    !    matr_a(2:3,2:3), 3, &
    !    matr_b(1:2,1:2), 3, &
    !    1.0d0,  &
    !    matr_c(2:3,1:2), 3 &
    !    )
    
end program test

subroutine general_dgemm_f(  &
    matr_a, rows_a, columns_a, start_row_a, len_row_a, start_column_a, len_column_a, opa, &
    matr_b, rows_b, columns_b, start_row_b, len_row_b, start_column_b, len_column_b, opb, &
    matr_c, rows_c, columns_c, start_row_c, len_row_c, start_column_c, len_column_c,      &
    alpha, beta &
)

    implicit none
    
    integer :: rows_a, columns_a, start_row_a, len_row_a, start_column_a, len_column_a
    integer :: rows_b, columns_b, start_row_b, len_row_b, start_column_b, len_column_b
    integer :: rows_c, columns_c, start_row_c, len_row_c, start_column_c, len_column_c

    integer :: k, lda, ldb;

    real*8 :: matr_a(rows_a,columns_a)
    real*8 :: matr_b(rows_b,columns_b)
    real*8 :: matr_c(rows_c,columns_c)
    real*8 :: alpha, beta

    character*1 :: opa, opb

    if (opa == 'N') then
        k = len_column_a
        lda = len_row_c
    else
        k = len_row_a
        lda = len_row_a
    endif

    if (opb == 'N') then
        ldb = len_row_b
    else
        ldb = len_column_c
    endif

    call dgemm(opa, opb, len_row_c, len_column_c, k, &
        alpha, &
        matr_a(start_row_a+1:len_row_a+start_row_a, start_column_a+1:len_column_a+start_column_a), lda, &
        matr_b(start_row_b+1:len_row_b+start_row_b, start_column_b+1:len_column_b+start_column_b), ldb, &
        beta,  &
        matr_c(start_row_c+1:len_row_c+start_row_c, start_column_c+1:len_column_c+start_column_c), len_row_c &
        )

end subroutine general_dgemm_f

! for the performance of O_V*V^{-1/2} in the generation of ri3fn 
!    (rest::molecule_io::prepare_ri3fn_for_ri_v_rayon)
subroutine special_dgemm_f_01 (  &
    ten3_a, x_a, y_a, z_a, start_x_a, len_x_a, i_y_a, start_z_a, len_z_a, &
    matr_b, rows_b, columns_b, start_row_b, len_row_b, start_column_b, len_column_b, &
    alpha, beta &
)

    implicit none
    
    integer :: x_a, y_a, z_a, start_x_a, len_x_a, i_y_a, start_z_a, len_z_a
    integer :: rows_b, columns_b, start_row_b, len_row_b, start_column_b, len_column_b

    integer :: i_state;

    real*8 :: ten3_a(x_a,y_a,z_a)
    real*8 :: matr_b(rows_b,columns_b)
    real*8 :: matr_c(rows_b,columns_b)
    real*8 :: alpha, beta


    do i_state = 1, y_a

        !call dgemm('N', 'N', len_x_a, len_column_b, len_z_a, &
        !    alpha, &
        !    ten3_a(start_x_a+1:len_x_a+start_x_a, i_state, start_z_a+1:len_z_a+start_z_a), len_x_a, &
        !    matr_b(start_row_b+1:len_row_b+start_row_b, start_column_b+1:len_column_b+start_column_b), len_z_a, &
        !    beta,  &
        !    ten3_a(start_x_a+1:len_x_a+start_x_a, i_state, start_z_a+1:len_z_a+start_z_a), len_x_a  &
        !    )

        matr_c(:,:) = ten3_a(start_x_a+1:len_x_a+start_x_a, i_state,  start_z_a+1:len_z_a+start_z_a)

        call dgemm('N', 'N', len_x_a, len_column_b, len_z_a, &
            alpha, &
            matr_c(:,:), len_x_a, &
            matr_b(start_row_b+1:len_row_b+start_row_b, start_column_b+1:len_column_b+start_column_b), len_z_a, &
            beta,  &
            matr_c(:,:), len_x_a  &
        )

        ten3_a(start_x_a+1:len_x_a+start_x_a, i_state,  start_z_a+1:len_z_a+start_z_a) = matr_c(:,:)

    end do

end subroutine special_dgemm_f_01
    


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


subroutine copy_mm(x_len, y_len, &
                   f_matr, f_x_len, f_y_len, f_x_start, f_y_start, &
                   t_matr, t_x_len, t_y_len, t_x_start, t_y_start)
    implicit none

    integer :: x_len, y_len
    integer :: f_x_len, f_y_len, f_x_start, f_y_start
    integer :: t_x_len, t_y_len, t_x_start, t_y_start

    real*8 :: f_matr(f_x_len, f_y_len)
    real*8 :: t_matr(t_x_len, t_y_len)

    t_matr(t_x_start+1:t_x_start+x_len, t_y_start+1:t_y_start+y_len) =  &
    f_matr(f_x_start+1:f_x_start+x_len, f_y_start+1:f_y_start+y_len)

end subroutine copy_mm

! copy a data block from a matrix into a rank-3 tensor
subroutine copy_mr(x1_len, x2_len, &
                   f_matr, f_x_len, f_y_len, f_x1_start, f_x2_start, &
                   t_ri, t_x_len, t_y_len, t_z_len, t_x1_start, t_x2_start, t_x3, mod)
    implicit none

    integer :: x1_len, x2_len
    integer :: f_x_len, f_y_len, f_x1_start, f_x2_start
    integer :: t_x_len, t_y_len, t_z_len, t_x1_start, t_x2_start, t_x3, mod

    real*8 :: f_matr(f_x_len, f_y_len)
    real*8 :: t_ri(t_x_len, t_y_len, t_z_len)

    if (mod .eq. 0) then
          t_ri(t_x1_start+1:t_x1_start+x1_len, t_x2_start+1:t_x2_start+x2_len, t_x3+1) =  &
        f_matr(f_x1_start+1:f_x1_start+x1_len, f_x2_start+1:f_x2_start+x2_len)
    else if (mod .eq. 1) then
          t_ri(t_x1_start+1:t_x1_start+x1_len, t_x3+1, t_x2_start+1:t_x2_start+x2_len) =  &
        f_matr(f_x1_start+1:f_x1_start+x1_len, f_x2_start+1:f_x2_start+x2_len)
    else if (mod .eq. 2) then
          t_ri(t_x3+1, t_x1_start+1:t_x1_start+x1_len, t_x2_start+1:t_x2_start+x2_len) = &
        f_matr(f_x1_start+1:f_x1_start+x1_len, f_x2_start+1:f_x2_start+x2_len)
    endif

end subroutine copy_mr

! copy a data block from a rank-3 tensor into a matrix
subroutine copy_rm(x1_len, x2_len, &
                   f_ri, f_x_len, f_y_len, f_z_len, f_x1_start, f_x2_start, f_x3, mod, &
                   t_matr, t_x_len, t_y_len, t_x1_start, t_x2_start)
    implicit none

    integer :: x1_len, x2_len
    integer :: t_x_len, t_y_len, t_x1_start, t_x2_start
    integer :: f_x_len, f_y_len, f_z_len, f_x1_start, f_x2_start, f_x3, mod

    real*8 :: t_matr(t_x_len, t_y_len)
    real*8 :: f_ri(f_x_len, f_y_len, f_z_len)

    if (mod .eq. 0) then
        t_matr(t_x1_start+1:t_x1_start+x1_len, t_x2_start+1:t_x2_start+x2_len) = &
          f_ri(f_x1_start+1:f_x1_start+x1_len, f_x2_start+1:f_x2_start+x2_len, f_x3+1) 
    else if (mod .eq. 1) then
        t_matr(t_x1_start+1:t_x1_start+x1_len, t_x2_start+1:t_x2_start+x2_len) =  &
          f_ri(f_x1_start+1:f_x1_start+x1_len, f_x3+1, f_x2_start+1:f_x2_start+x2_len)
    else if (mod .eq. 2) then
        t_matr(t_x1_start+1:t_x1_start+x1_len, t_x2_start+1:t_x2_start+x2_len) = &
          f_ri(f_x3+1, f_x1_start+1:f_x1_start+x1_len, f_x2_start+1:f_x2_start+x2_len)
    endif

end subroutine copy_rm

subroutine copy_rr(x1_len, x2_len, x3_len,&
                   f_ri, f_x1_len, f_x2_len, f_x3_len, f_x1_start, f_x2_start, f_x3_start,&
                   t_ri, t_x1_len, t_x2_len, t_x3_len, t_x1_start, t_x2_start, t_x3_start)
    implicit none

    integer :: x1_len, x2_len, x3_len
    integer :: f_x1_len, f_x2_len, f_x3_len, f_x1_start, f_x2_start, f_x3_start
    integer :: t_x1_len, t_x2_len, t_x3_len, t_x1_start, t_x2_start, t_x3_start

    real*8 :: f_ri(f_x1_len, f_x2_len, f_x3_len)
    real*8 :: t_ri(t_x1_len, t_x2_len, t_x3_len)

        t_ri(t_x1_start+1:t_x1_start+x1_len, &
             t_x2_start+1:t_x2_start+x2_len, &
             t_x3_start+1:t_x3_start+x3_len) =  &
        f_ri(f_x1_start+1:f_x1_start+x1_len, &
             f_x2_start+1:f_x2_start+x2_len, &
             f_x3_start+1:f_x3_start+x3_len)

end subroutine copy_rr