T=zeros(24,3);
T(1,:)=[1 2 3];
T(2,:)=[1 3 4];
T(3,:)=[1 4 5];
T(4,:)=[6 1 5];
T(5,:)=[6 7 1];
T(6,:)=[7 2 1];
T(7,:)=[7 10 2];
T(8,:)=[10 11 2];
T(9,:)=[11 12 2];
T(10,:)=[2 12 13];
T(11,:)=[2 13 3];
T(12,:)=[3 14 4];
T(13,:)=[3 13 14];
T(14,:)=[4 17 5];
T(15,:)=[4 14 17];
T(16,:)=[6 5 9];
T(17,:)=[5 17 18];
T(18,:)=[5 18 19];
T(19,:)=[9 5 19];
T(20,:)=[8 7 6];
T(21,:)=[8 6 9];
T(22,:)=[15 7 8];
T(23,:)=[16 7 15];
T(24,:)=[16 10 7];
Undertermined=[12 13 14 15 16 10 11]; % circles on the boundary with undertermined surround angle
Determined_interior=[1 2 3 4 5 6 7 ]; % interior circles
Determined_boundary=[17 18 9 8]; % circle on the boundary with dertermined surround angle pi
Determined_corner=19; % circle on the boundary with dertermined surround angle pi/2
R=zeros(19,2);
%Set boundary circle radii
for i=1:length(Undertermined)
    %R(Undertermined(i),1)=1;
    R(Undertermined(i),1)=0.5 + 0.5*rand(1);
    R(Undertermined(i),2)=0;
    
end
%Randomly set interior circle radii
for i=1:length(Determined_interior)
    R(Determined_interior(i),1)=0.5 + 0.5*rand(1);
    R(Determined_interior(i),2)=2*pi;
end

for i=1:length(Determined_boundary)
    R(Determined_boundary(i),1)=0.5 + 0.5*rand(1);
    R(Determined_boundary(i),2)=pi;
end

R(Determined_corner,1)=0.5 + 0.5*rand(1);
R(Determined_corner,2)=pi/2;
