%% get the lines and maximum radius point
roiImage = imread('newColorDop.JPG'); 
img = imshow(roiImage);
img.AlphaData = 0.5;
hold on;
% Get 5 mouse clicks from the user in the current figure
disp("mark the 5 co-ordinates");
[x,y] = ginput(5);
%% -----Calculate the beam centre and get the angle of the beam -----------

m1 = ( y(2) - y(1) )/( x(2) - x(1) ); % slopes of the two lines
m2 = ( y(4) - y(3))/( x(4) - x(3) );
x(6) = ( y(4) - y(2) + m1*x(2) - m2*x(4) )/(m1 - m2); % x beam centre
y(6) = m1*( x(6) - x(2) ) + y(2); % y beam centre
x(7) = x(6)-80; %perpendicular line from the beam centre
y(7) = y(6);
% Draw the lines that the points represent
line(x(1:2), y(1:2));
line(x(3:4), y(3:4));
line(x(6:7), y(6:7));
% Define the two vectors
v1 = [x(2) - x(1), y(2) - y(1)];
v2 = [x(4) - x(3), y(4) - y(3)];
v3 = [x(7) - x(6), y(7) - y(6)];
% Compute the angle from v1 to v2
theta = acosd(dot(v1, v2) / (norm(v1) * norm(v2)) ); % angle subtended by beam 
%disp(theta);
disp("calculated theta(not used) and beam centre");
%------------------------got theta and beam centre------------------------------- 

%% Plot the largest arc

% hold on;
xCenter = x(6);
yCenter = y(6);
% tetaI = acosd(dot(v1, v3) / (norm(v1) * norm(v3)) );
% tetaF =  acosd(dot(v3, v2) / (norm(v3) * norm(v2)) );
% disp(tetaI);
% disp(tetaF);
% teta =180 + tetaI : 0.01 : 180+ tetaF ;
% %disp(length(teta));
% rmax = sqrt( ( x(6) - x(5) )^2 + ( y(6) - y(5) )^2); %maximum radius distance
% radius = rmax;  % final maximum radius
% %disp(radius);
% xr = round(radius * cosd(teta) + xCenter);
% yr = round(radius * sind(-teta) + yCenter);
% plot(xr, yr,'r');
% %disp(xr);
% disp("plotted radius...press any key to continue");
% 
% pause;
% hold off;

%% Preprocess the image
%roiImage = imread('newColorDop.JPG');
%imshow(roiImage);
% medImage = medfilt3(roiImage);
% gaussImage = imgaussfilt(medImage,2.8); % SD = 8 pixels

%roiImage = gaussImage;
%imwrite(roiImage,'newColorDop_pre.JPG');
%disp("the image is preprocessed");
%imwrite(gaussImage,'colorImage_10pre.JPG');
%imshowpair(gaussImage,medImage,'montage');

%% 
disp("crop the color bar");
Image = imread('newColorDop.JPG'); 
imshow(Image);
h = imrect;
pause;

position = getPosition(h); 
croppedColorBar = imcrop(Image,position);
hold off;
%figure; 
%imshow(croppedColorBar);
color_bar = squeeze(uint8((mean(croppedColorBar,2))));% color_bar contains a single rgb colour
%disp(color_bar);

%------got the color bar as 2D matrix with r g b as columns----------------

%% Map rgb value with the velocity.
cmap = double(color_bar); 

mindata = -0.84; %set the value from the image
maxdata = 0.84; 
%% Map the velocities and calculate U(x,y)

%roiImage = imread('colorImage_10.JPG'); 
%imshow(roiImage);
%h = imrect;
%pause; 
%position = getPosition(h); 
%ImageROI = imcrop(roiImage,position);
ImageROI = roiImage;
[rows,cols,~] = size(ImageROI);
%figure; 
%imshow(ImageROI);
disp("Calculating U(x,y) for the whole image");
U = zeros(rows,cols);
for locY = 1:rows
    for locX = 1:cols
  
        rvalue = double(ImageROI(locY,locX,1));
        gvalue = double(ImageROI(locY,locX,2));
        bvalue = double(ImageROI(locY,locX,3));
        rgbInBar = [rvalue,gvalue,bvalue]; 

        repeat = repmat(rgbInBar,length(cmap),1);
        compare = cmap - repeat ;
        compare = compare.^2;
        comparetotal = sum(compare,2);
        [valuediff,rowLoc] = min(sqrt(comparetotal));  %the minimumn euclidean
                                                        % distance

        currentPointVelocity = mindata+(maxdata-mindata)*rowLoc/length(cmap);%U(x,y)
        U(locY,locX) = currentPointVelocity;
    end
end
disp("calculated U(x,y)");
%disp(size(U));
%--------------------got U(x,y)--------------------------------------------

%% Storing the Radial velocity component in a matrix
% rmaxi = round(radius);
% tetamax = length(teta);
% Vr = zeros([rmaxi tetamax]);
% %disp(size(Vr));
% Vazi = Vr;
% 
% %disp(xCenter);
% %disp(yCenter);
%  for r = 1:1:rmaxi 
%      
%    for t = 1:1:tetamax
%           trad = tetaI + (t - 1)*(tetaF - tetaI)/(tetamax - 1);
%             xv =round( r * cos(trad) + xCenter );
%             %disp(xr);
%             yv =round( r * sin(-trad) + yCenter ); %plot co-ordinates are different from matrix'  
%             %fprintf("r = %d, (%d,%d) \n",r,xr,yr);
%             if(xv>0)
%                 Vr(r,t) = U(yv,xv);
%             end
%    end
%  end
% % disp(size(Vr));
% 
%  disp("Calculated V(r,theta)");
%-------------------------Vr matrix is obtained (correct)----------------------------
%% Get the boundary of the wall
disp("Draw the wall boundary");
image = imread('newColorDop.JPG'); 
imshow(roiImage);
h = imfreehand;
pause;
position = getPosition(h); 

%ImageROI = imcrop(roiImage,position);
%[rows,cols,~] = size(ImageROI);
%disp(size(ImageROI));
%disp(position);
[Nofpts,~] = size(position); 
%BdryXY = zeros(Nofpts,3);
BdryXY = position;
for n = 1:Nofpts
    BdryXY(n,3) = round(sqrt( ( x(6) - BdryXY(n,1) )^2 + ( y(6) - BdryXY(n,2) )^2));
end
BdryXYSorted = sortrows(BdryXY,[3 1]); 
disp("Sorted Boundary points are obtained");
%disp(BdryXYSorted(:,3));
%-------------- soreted boundary points ar obtained------------------------
%% Get theta - and theta +
rIni =  BdryXYSorted(1,3);
%disp(rIni);
rFin =  BdryXYSorted(Nofpts,3);
%disp(rFin);
theta = zeros([rFin-rIni,7]);
[smax,~] = size(BdryXYSorted);
s = 1;i = 1;
for r= 1 : 1 : rFin - rIni
if s < smax 
      checkr = BdryXYSorted(s,3);
      s1 = s;
      s = s+1;
      s2 = s;
    disp(s);
    while (BdryXYSorted(s,3) == checkr) &&  (s < smax) 
        if(BdryXYSorted(s2,1)-BdryXYSorted(s1,1) > 8)
            theta(r,:) = [BdryXYSorted(s1,1),BdryXYSorted(s1,2),BdryXYSorted(s2,1),BdryXYSorted(s2,2),BdryXYSorted(s2,3),0,0];         
        end
        s = s+1;
        s2 = s;
    end
    %disp(theta(r,:));
end
end
%disp(theta);
%disp(BdryXYSorted);
disp("End points of the arc are obtained");

v3 = [x(7) - x(6), y(7) - y(6)];
for i = 1 : 1 : rFin - rIni 
if(theta(i,1)>0)
    v4 = [theta(i,1) - x(6), theta(i,2) - y(6)];
    v5 = [theta(i,3) - x(6), theta(i,4) - y(6)];
    theta(i,6) = acosd(dot(v3, v4) / (norm(v3) * norm(v4)) );
    theta(i,7) = acosd(dot(v5, v3) / (norm(v3) * norm(v5)) );
end
end 
disp("theta- and theta+ are calculated");
disp(size(theta));
%---------- obtained theta matrix cotaining radius,theta- and theta+-------
%% Store the VrROI from U(x,y) as a matrix
t = theta(:,6);
%disp(theta(:,6));
t(t==0) = inf;
[tetamin,~] = min(t);
[tetamax,~] = max(theta(:,7));
disp(tetamin);
disp(tetamax);
tetaLen = round(tetamax - tetamin);
VrROI = zeros(rFin - rIni,10*tetaLen);
LNZ = find(theta(:,1)>0);
rf = LNZ(end);
 imshow('newColorDop.JPG');
for r = 1:1:(rFin - rIni) 
    if(theta(r,1) > 0)
        for t = 1:1:10*tetaLen
            rad = theta(r,5);
            tdeg = tetamin + tetaLen*(t - 1)/(10*tetaLen - 1) + 180;
            %disp(tdeg);
            if(tdeg - 180 >= theta(r,6) && tdeg - 180 <= theta(r,7))
                xv =round( rad * cosd(tdeg) + xCenter );
                %disp(xr);
                 %disp(yv);
                yv =round( rad * sind(-tdeg) + yCenter ); %plot co-ordinates are different from matrix'  
                %fprintf("r = %d, (%d,%d) \n",r,xr,yr);

                 VrROI(r,t) = U(yv,xv);
%                   axis on
%                   
%                   
%                   hold on;
% %                   % Plot cross at row 100, column 50
%                   plot(xv,yv, 'y+', 'MarkerSize', 1, 'LineWidth', 0.5);
            end
        end
    end   
    if(theta(r,1) == 0 && r>1 &&r < rf+1)
        disp("excuse me");
      
        theta(r,:)=[1,0,0,0,theta(r-1,5)+1,theta(r-1,6),theta(r-1,7)];
        for t = 1:1:10*tetaLen
            rad = theta(r,5);
            tdeg = tetamin + tetaLen*(t - 1)/(10*tetaLen - 1) + 180;
            %disp(tdeg);
            if(tdeg - 180 >= theta(r,6) && tdeg - 180 <= theta(r,7))
                xv =round( rad * cosd(tdeg) + xCenter );
                %disp(xr);
                 %disp(yv);
                yv =round( rad * sind(-tdeg) + yCenter ); %plot co-ordinates are different from matrix'  
                %fprintf("r = %d, (%d,%d) \n",r,xr,yr);

                 VrROI(r,t) = U(yv,xv);
%                  axis on
%                  hold on;
%                   % Plot cross at row 100, column 50
%                  plot(xv,yv, 'y+', 'MarkerSize', 1, 'LineWidth', 0.5);
            end
        end
    end   
end
               
disp("obtained VrROI(polar)");



 %% Calculating VaziNegative
 
 VaziN = VrROI;
 
 [~,gVr] = gradient(VrROI);  % r varies with row,vertical direction.
 
 Integrand = VrROI;
for r = 1:1: rFin - rIni
    if(theta(r,1)>0)
        Integrand(r,:) = -theta(r,5).*gVr(r,:) - VrROI(r,:);
        cf =round( 1 + (10*tetaLen - 1)*(theta(r,7) - tetamin)/tetaLen );%convert angle interms of rows
        ci = round( 1 + (10*tetaLen - 1)*(theta(r,6) - tetamin)/tetaLen );
        %disp(cf);
        %disp(r);
        if(cf >  10*tetaLen)
            cf = tetaLen;
        end   
      
        VaziN(r,ci+1:cf) = cumtrapz(Integrand(r,ci+1:cf)); % along theta i.e., along rows. varies with columns
                        %include as many columns you want in Integrand
    end
end
%disp(VaziN);
disp("VaziN is calculated...");

%------------obtained VaziN -----------------------------------------------

%% Calculating VaziPositive 

VaziP = flip(VrROI,2);

Integrand = VrROI;
for r = 1:1: rFin - rIni
    if(theta(r,1)>0)
        Integrand(r,:) = theta(r,5).*gVr(r,:) + VrROI(r,:);
        cf =round( 1 + (10*tetaLen - 1)*(theta(r,7) - tetamin)/tetaLen );%convert angle interms of rows
        cin = round( 1 + (10*tetaLen - 1)*(theta(r,6) - tetamin)/tetaLen );
     
        
        %disp(r);
        if(cf >  10*tetaLen)
            cf = 10*tetaLen;
        end   
        ci = 10*tetaLen + 1 - cf;
        cf = 10*tetaLen + 1 - cin; 
      
        VaziP(r,ci+1:cf) = cumtrapz(Integrand(r,ci+1:cf)); % along theta i.e., along rows. varies with columns
                        %include as many columns you want in Integrand
    end
end
VaziP = -VaziP;
VaziP = flip(VaziP,2);
disp(VaziP);
disp("VaziP is calculated...");

%------------obtained VaziP -----------------------------------------------

%% Calculating Vazi from the weights (have to fix this)

% Vazi = VaziP;
% 
% for r = 1 : 1 : rFin - rIni
%     if(theta(r,1)>0)
%         Integrand(r,:) = r.*gVr(r,:) + VrROI(r,:);
%         cf =round( 1 + (10*tetaLen - 1)*(theta(r,7) - tetamin)/tetaLen );%convert angle interms of rows
%         cin = round( 1 + (10*tetaLen - 1)*(theta(r,6) - tetamin)/tetaLen );
%      
%         
%         %disp(r);
%         if(cf >  10*tetaLen)
%             cf = 10*tetaLen;
%         end   
%         ci = 10*tetaLen + 1 - cf;
%         cf = 10*tetaLen + 1 - cin; 
%       
%         VaziP(r,ci+1:cf) = cumtrapz(Integrand(r,ci+1:cf)); % along theta i.e., along rows. varies with columns
%                         %include as many columns you want in Integrand
%     end
% end
%% Back to Cartesian co-ordinates (VrROI and Vazi)
VrROIxy = zeros(size(U));
Vazixy = zeros(size(U));
VaziNxy = zeros(size(U));
VaziPxy = zeros(size(U));
Confuse = zeros(size(U));
for r = 1:1:(rFin - rIni) 
    if(theta(r,1) > 0)
        disp(r);
        for t = 1:1:10*tetaLen
            
            rad = theta(r,5);
            tdeg = tetamin + tetaLen*(t - 1)/(10*tetaLen - 1) + 180;
            %disp(tdeg);
            if(tdeg - 180 >= theta(r,6) && tdeg - 180 <= theta(r,7))
                xv =round( rad * cosd(tdeg) + xCenter );
                %disp(xr);
                 %disp(yv);
                yv =round( rad * sind(-tdeg) + yCenter ); %plot co-ordinates are different from matrix'  
                %fprintf("r = %d, (%d,%d) \n",r,xr,yr);
                Confuse(r,t) = yv;
                VrROIxy(yv,xv) = VrROI(r,t);
                VaziNxy(yv,xv) = VaziN(r,t); 
                VaziPxy(yv,xv) = VaziP(r,t); 
                %disp(VrROI(r,t));
%                  axis on
%                  hold on;
%                   % Plot cross at row 100, column 50
%                  plot(xv,yv, 'y+', 'MarkerSize', 3, 'LineWidth', 2);
            end      
        end
    end   
end
  %disp(VrROIxy);    
  disp("obtained VrROIxy,VaziNxy,VaziPxy");
  
  %% Calculating weight matrix
  [imax,jmax] = size(VrROIxy);
  W = zeros(size(U));
%   imshow('newColorDop_pre.JPG');
  for i = 1:1:imax
     % L = sum(VrROIxy(i,:) > 0);
      Nz   = find(VrROIxy(i,:) > 0);
      if(isempty(Nz)== 0)
          FnzCol = Nz(1);
          LnzCol = Nz(end);
          for j = FnzCol:1:LnzCol
              W(i,j) = (j-FnzCol)/(LnzCol - FnzCol);  
%                  axis on
%                  hold on;
%                    % Plot cross
%                  plot(j,i, 'y+', 'MarkerSize', 3, 'LineWidth', 2);
          end
      end
    
  end
  disp(W);
  disp("Calculated weight matrix");
  
  %% Calculating Vazixy
  Vazixy = W.*VaziNxy + (1-W).*VaziPxy;
  disp("Calculated Vazixy from weight matrix");
  %% Check Vazi and Vr
 

 %% ------Plotting the velocity vectors on the image------------------------
disp("plotting the vectors");

figure;
imshow(roiImage,[]);
hold on;
[Nx, Ny,~] = size(roiImage);
xidx = 1:1:Nx;
yidx = 1:1:Ny;
[X,Y] = meshgrid(xidx,yidx);
% % % jump = 2;
% % % Y = Y(:,1:jump:end);
% % % X = X(:,1:jump:end);
% % % VrROIxy = VrROIxy(:,1:jump:end);
% % % Vazixy = Vazixy(:,1:jump:end);

sz = size(Vazixy); 
Z = zeros(sz);
%g = hgtransform;
%quiver(x,y,u,v,'Parent',g)

%compass(Z,VrROIxy);

q = quiver(Y',X',Z,VrROIxy,2);
%disp(r);
%set(g,'Matrix',makehgtform('zrotate',pi/3))
%q = quiver(Y',X',U,Z,2);
%q = quiver(U,V);
q.Color = 'yellow';
q.MaxHeadSize = 1.5;
q.MarkerSize = 10;
disp("plotted vectors");


%% Plotting VrROIxy in the direction of beam
[row,col] = size(U);
theeeta = zeros(size(U));
UVr = theeeta;
VVr = theeeta;
for i = 1:row
    for j = 1:col
    vpoint = [j - x(6), i - y(6)];
    angle = acosd(dot(v3, vpoint) / (norm(v3) * norm(vpoint)) );
    theeeta(i,j) = angle;
    end
end
%%
UVr = VrROIxy.*cosd(theeeta);
VVr = VrROIxy.*sind(theeeta);
figure;
imshow(roiImage,[]);
hold on;
 
q = quiver(Y',X',-UVr,VVr,2);
%disp(r);
%set(g,'Matrix',makehgtform('zrotate',pi/3))
%q = quiver(Y',X',U,Z,2);
%q = quiver(U,V);
q.Color = 'yellow';
q.MaxHeadSize = 1.5;
q.MarkerSize = 10;
disp("plotted VrROIxy vectors along the beam direction");
%% Plotting Vazixy perpendicular to the beam direction

UVr = Vazixy.*cosd(90 - theeeta);
VVr = Vazixy.*sind(90 - theeeta);
%figure;
%imshow(roiImage,[]);
hold on;
 
q = quiver(Y',X',-UVr,-VVr,2);
%disp(r);
%set(g,'Matrix',makehgtform('zrotate',pi/3))
%q = quiver(Y',X',U,Z,2);
%q = quiver(U,V);
q.Color = 'red';
q.MaxHeadSize = 1.5;
q.MarkerSize = 10;
disp("plotted VrROIxy vectors along the beam direction");




