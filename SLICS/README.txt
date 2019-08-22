The files are in binary format, float(4), storing (x,y,x,vx,vy,vz) on each particle, with a header containing 12 floats that you can ignore.

The best way to look at them in terminal is with 'od':

>od -f -N 480 -A n  -w24 -j48 0.042xv0.dat
       760.38696        763.2298       3.1335907        12.76199        8.769131       10.085842
       739.45886       761.63873       3.1369312       13.478473        8.516876        9.107691
        735.5625       764.83215       3.6035395       12.804067        8.418052        8.031185
        759.4267        764.1946       5.5938125       15.364888        9.273083        9.904539
        758.6591        763.2264       4.1066985       13.914708        8.650762        9.660725
       731.93854        764.3814       3.5195658       15.685057         8.98524        9.953589
        729.1017        761.9736       3.2009735        14.21476       6.6003227        9.623456
        728.9026       761.70886       3.2896674       15.725218        7.502788       10.468324
        728.8798       761.92096         3.07734       16.591621        7.926739        9.234021
       731.34973        764.4804       7.2060523        14.09263        6.360201       10.582818
        727.9892        764.1572       6.2821927       16.948246       7.0567384        9.604617

You can unpack just x,y,z into a ascii file with:

od -f -N 480 -A n  -w24 -j48 0.042xv0.dat | gawk {'print $1 "  " $2 "  " $3}' > 0.042xv0_ascii.dat

Or I know you can read the binary itself from python with something like:

with open(fname1, 'rb') as f1:
        data_bin = np.fromfile(f1, dtype=np.float32)

data_bin[0:20]
array([ 2.8061820e-37,  9.5465004e-01,  2.9512144e+01, -3.4878576e+00,
        4.0595617e-42,  1.7161643e-02,  5.1614386e-04,  6.9625117e-02,
        3.3631163e-44,  2.6624671e-44,  2.6624671e-44,  8.0000000e+00,
        7.6038696e+02,  7.6322980e+02,  3.1335907e+00,  1.2761990e+01,
        8.7691307e+00,  1.0085842e+01,  7.3945886e+02,  7.6163873e+02],
      dtype=float32)

(The first 12 entries are to be trashed, but the rest match the above)

Then throw away the first 12 entries and reshape the rest as a 6xNp array.

data_bin_noheader = data_bin[12:]

numel = np.shape(data_bin_noheader)
numel[0]/6
den_final = np.reshape(data_bin_noheader, [int(numel[0]/6), 6])

Here is the logic:

The simulations were run on 64 MPI tasks, each working on a sub volume of the box.
The particle data are stored assuming their local sub-volume 'cell' coordinate, hence every coordinate entry of every particle should be bracketed in the range [0 - 3072]/4 = [0-768]. These coordinates must first be shifted to their global position, then transformed from cell units to Mpc/h.

1- local to global shift

! i = label that runs from 0 to 63. That correspond to the label in the particle file name.
! So for every value of i, run the code snippet below, and solve for node_coords, a vector with 3 entries, corresponding ! to the shift in (x,y,z) to apply to shift from local coordinate to global coordinate.


nodes_dim = 4  ! that's the number of MPI task per dimension. 4^3 = 64
ncc = 768         ! an integer
rnc = 3070.0     ! a float

do k1=1,nodes_dim
   do j1=1,nodes_dim
      do i1=1,nodes_dim
         if (i == (i1-1)+(j1-1)*nodes_dim+(k1-1)*nodes_dim**2)  then
            node_coords(:)=(/(i1-1),(j1-1),(k1-1)/)
         endif
      enddo
   enddo
enddo

! apply the shift here:

xv_global(1,:)=modulo(xv_file(1,:)+node_coords(1)*ncc,rnc)
xv_global(2,:)=modulo(xv_file(2,:)+node_coords(2)*ncc,rnc)
xv_global(3,:)=modulo(xv_file(3,:)+node_coords(3)*ncc,rnc)

! now all particles from the file have their position in global grid cell units.
! you can check that they all live within the range [0-3072], perhaps verify this with a histogram or some visualisation
! of a randomly extracted subset?

! Grid cell units to Mpc/h units:

xv_mpc_over_h(1:3,:) = xv_global(1:3,:)*505./3072.
