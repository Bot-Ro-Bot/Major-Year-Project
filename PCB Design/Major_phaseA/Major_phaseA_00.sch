EESchema Schematic File Version 4
EELAYER 30 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title ""
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L Amplifier_Instrumentation:AD620 U1
U 1 1 5E47C174
P 2458 1400
F 0 "U1" H 2902 1446 50  0000 L CNN
F 1 "AD620" H 2902 1355 50  0000 L CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 2458 1400 50  0001 C CNN
F 3 "https://www.analog.com/media/en/technical-documentation/data-sheets/AD620.pdf" H 2458 1400 50  0001 C CNN
	1    2458 1400
	1    0    0    -1  
$EndComp
$Comp
L oppo:op37G U2
U 1 1 5E47D3F8
P 4098 1357
F 0 "U2" H 4123 1788 50  0000 C CNN
F 1 "op37G" H 4123 1697 50  0000 C CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 4098 1357 50  0001 C CNN
F 3 "" H 4098 1357 50  0001 C CNN
	1    4098 1357
	1    0    0    -1  
$EndComp
$Comp
L oppo:op37G U3
U 1 1 5E47DACF
P 5784 1342
F 0 "U3" H 5809 1773 50  0000 C CNN
F 1 "op37G" H 5809 1682 50  0000 C CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 5784 1342 50  0001 C CNN
F 3 "" H 5784 1342 50  0001 C CNN
	1    5784 1342
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R1
U 1 1 5E47E754
P 1787 1402
F 0 "R1" H 1855 1448 50  0000 L CNN
F 1 "100 Ohm" H 1855 1357 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 1787 1402 50  0001 C CNN
F 3 "~" H 1787 1402 50  0001 C CNN
	1    1787 1402
	1    0    0    -1  
$EndComp
Wire Wire Line
	1787 1502 2058 1502
Wire Wire Line
	2058 1502 2058 1500
Wire Wire Line
	1787 1302 2059 1302
Wire Wire Line
	2059 1302 2059 1300
Wire Wire Line
	2059 1300 2058 1300
$Comp
L Connector:Conn_01x03_Male J2
U 1 1 5E47FB19
P 2473 594
F 0 "J2" V 2535 738 50  0000 L CNN
F 1 "Conn_01x03_Male" V 2626 738 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 2473 594 50  0001 C CNN
F 3 "~" H 2473 594 50  0001 C CNN
	1    2473 594 
	0    1    1    0   
$EndComp
Wire Wire Line
	2373 794  2373 822 
Wire Wire Line
	2373 1100 2358 1100
Wire Wire Line
	2558 1700 2508 1700
Wire Wire Line
	2473 1700 2473 917 
Wire Wire Line
	2358 1700 2358 1729
Wire Wire Line
	2358 1729 2625 1729
Wire Wire Line
	2625 848  2573 848 
Wire Wire Line
	2573 848  2573 794 
$Comp
L power:GND #PWR02
U 1 1 5E482B6D
P 2088 885
F 0 "#PWR02" H 2088 635 50  0001 C CNN
F 1 "GND" V 2093 757 50  0000 R CNN
F 2 "" H 2088 885 50  0001 C CNN
F 3 "" H 2088 885 50  0001 C CNN
	1    2088 885 
	0    1    1    0   
$EndComp
$Comp
L power:+9V #PWR03
U 1 1 5E483466
P 2092 782
F 0 "#PWR03" H 2092 632 50  0001 C CNN
F 1 "+9V" V 2107 910 50  0000 L CNN
F 2 "" H 2092 782 50  0001 C CNN
F 3 "" H 2092 782 50  0001 C CNN
	1    2092 782 
	0    -1   -1   0   
$EndComp
$Comp
L power:-9V #PWR01
U 1 1 5E483BA9
P 2084 984
F 0 "#PWR01" H 2084 859 50  0001 C CNN
F 1 "-9V" V 2099 1112 50  0000 L CNN
F 2 "" H 2084 984 50  0001 C CNN
F 3 "" H 2084 984 50  0001 C CNN
	1    2084 984 
	0    -1   -1   0   
$EndComp
Wire Wire Line
	2092 782  2092 822 
Wire Wire Line
	2092 822  2373 822 
Connection ~ 2373 822 
Wire Wire Line
	2373 822  2373 1066
Wire Wire Line
	2087 885  2088 885 
Connection ~ 2473 885 
Wire Wire Line
	2473 885  2473 794 
Connection ~ 2088 885 
Wire Wire Line
	2088 885  2473 885 
Wire Wire Line
	2084 984  2625 984 
Wire Wire Line
	2625 848  2625 984 
Connection ~ 2625 984 
Wire Wire Line
	2625 984  2625 1171
$Comp
L Connector:Conn_01x03_Male J1
U 1 1 5E485FB5
P 1092 1378
F 0 "J1" H 1200 1659 50  0000 C CNN
F 1 "Conn_01x03_Male" H 1200 1568 50  0000 C CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 1092 1378 50  0001 C CNN
F 3 "~" H 1092 1378 50  0001 C CNN
	1    1092 1378
	1    0    0    -1  
$EndComp
Wire Wire Line
	1292 1278 1346 1278
Wire Wire Line
	1699 1278 1699 1224
Wire Wire Line
	1699 1224 2058 1224
Wire Wire Line
	2058 1224 2058 1200
Wire Wire Line
	1292 1378 1374 1378
Wire Wire Line
	1698 1378 1698 1617
Wire Wire Line
	1698 1617 2058 1617
Wire Wire Line
	2058 1617 2058 1600
Wire Wire Line
	1292 1478 1399 1478
Wire Wire Line
	1642 1478 1642 1811
Wire Wire Line
	1642 1811 2508 1811
Wire Wire Line
	2508 1811 2508 1700
Connection ~ 2508 1700
Wire Wire Line
	2508 1700 2473 1700
Text GLabel 1082 1621 0    50   Input ~ 0
Electrode1a
Text GLabel 1077 1767 0    50   Input ~ 0
Electrode2a
Text GLabel 1301 2048 0    50   Input ~ 0
reference_Electrode
Wire Wire Line
	1082 1621 1346 1621
Wire Wire Line
	1346 1621 1346 1278
Connection ~ 1346 1278
Wire Wire Line
	1346 1278 1699 1278
Wire Wire Line
	1077 1767 1374 1767
Wire Wire Line
	1374 1767 1374 1378
Connection ~ 1374 1378
Wire Wire Line
	1374 1378 1698 1378
Connection ~ 1399 1478
Wire Wire Line
	1399 1478 1642 1478
$Comp
L Device:CP_Small C1
U 1 1 5E48F57C
P 3039 1399
F 0 "C1" V 3264 1399 50  0000 C CNN
F 1 "0.1 uF" V 3173 1399 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 3039 1399 50  0001 C CNN
F 3 "~" H 3039 1399 50  0001 C CNN
	1    3039 1399
	0    -1   -1   0   
$EndComp
$Comp
L Device:CP_Small C2
U 1 1 5E49018B
P 3397 1400
F 0 "C2" V 3622 1400 50  0000 C CNN
F 1 "01 uF" V 3531 1400 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 3397 1400 50  0001 C CNN
F 3 "~" H 3397 1400 50  0001 C CNN
	1    3397 1400
	0    -1   -1   0   
$EndComp
$Comp
L Device:R_Small_US R2
U 1 1 5E490CED
P 3221 1585
F 0 "R2" H 3289 1631 50  0000 L CNN
F 1 "1.5 MOhm" H 3289 1540 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3221 1585 50  0001 C CNN
F 3 "~" H 3221 1585 50  0001 C CNN
	1    3221 1585
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R3
U 1 1 5E49138A
P 3647 1553
F 0 "R3" H 3715 1599 50  0000 L CNN
F 1 "1.5 MOhm" H 3715 1508 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3647 1553 50  0001 C CNN
F 3 "~" H 3647 1553 50  0001 C CNN
	1    3647 1553
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R5
U 1 1 5E491B38
P 3858 955
F 0 "R5" V 3653 955 50  0000 C CNN
F 1 "3.3 KOhm" V 3744 955 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3858 955 50  0001 C CNN
F 3 "~" H 3858 955 50  0001 C CNN
	1    3858 955 
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R9
U 1 1 5E496A1B
P 5747 818
F 0 "R9" V 5542 818 50  0000 C CNN
F 1 "5 KOhm" V 5633 818 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5747 818 50  0001 C CNN
F 3 "~" H 5747 818 50  0001 C CNN
	1    5747 818 
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R8
U 1 1 5E496E2C
P 5373 640
F 0 "R8" H 5441 686 50  0000 L CNN
F 1 "10 KOhm" H 5441 595 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5373 640 50  0001 C CNN
F 3 "~" H 5373 640 50  0001 C CNN
	1    5373 640 
	1    0    0    -1  
$EndComp
$Comp
L Device:CP_Small C4
U 1 1 5E49A511
P 5709 1823
F 0 "C4" V 5934 1823 50  0000 C CNN
F 1 "0.1 uF" V 5843 1823 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 5709 1823 50  0001 C CNN
F 3 "~" H 5709 1823 50  0001 C CNN
	1    5709 1823
	0    -1   -1   0   
$EndComp
$Comp
L Device:CP_Small C3
U 1 1 5E49AE10
P 5453 1509
F 0 "C3" H 5365 1463 50  0000 R CNN
F 1 "0.1 uF" H 5365 1554 50  0000 R CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 5453 1509 50  0001 C CNN
F 3 "~" H 5453 1509 50  0001 C CNN
	1    5453 1509
	-1   0    0    1   
$EndComp
$Comp
L Device:R_Small_US R7
U 1 1 5E49B342
P 5256 1387
F 0 "R7" V 5051 1387 50  0000 C CNN
F 1 "16 KOhm" V 5142 1387 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5256 1387 50  0001 C CNN
F 3 "~" H 5256 1387 50  0001 C CNN
	1    5256 1387
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R6
U 1 1 5E49B704
P 4781 1379
F 0 "R6" V 4576 1379 50  0000 C CNN
F 1 "16 KOhm" V 4667 1379 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 4781 1379 50  0001 C CNN
F 3 "~" H 4781 1379 50  0001 C CNN
	1    4781 1379
	0    1    1    0   
$EndComp
Wire Wire Line
	2858 1400 2939 1400
Wire Wire Line
	2939 1400 2939 1399
Wire Wire Line
	3297 1399 3297 1400
Wire Wire Line
	3221 1485 3221 1399
Wire Wire Line
	3139 1399 3221 1399
Connection ~ 3221 1399
Wire Wire Line
	3221 1399 3297 1399
Wire Wire Line
	3497 1400 3647 1400
Wire Wire Line
	3848 1400 3848 1407
Wire Wire Line
	3647 1453 3647 1400
Connection ~ 3647 1400
Wire Wire Line
	3647 1400 3848 1400
Wire Wire Line
	3663 955  3758 955 
Wire Wire Line
	3958 955  4440 955 
Wire Wire Line
	4440 955  4440 1357
Wire Wire Line
	3848 1307 3663 1307
Wire Wire Line
	3663 739  3663 955 
Connection ~ 3663 955 
Wire Wire Line
	3663 955  3663 1307
Wire Wire Line
	4681 1357 4681 1379
Wire Wire Line
	4398 1357 4440 1357
Connection ~ 4440 1357
Wire Wire Line
	4440 1357 4681 1357
Wire Wire Line
	4881 1379 5029 1379
Wire Wire Line
	5156 1379 5156 1387
Wire Wire Line
	5356 1387 5453 1387
Wire Wire Line
	5453 1387 5453 1388
Wire Wire Line
	5453 1388 5534 1388
Wire Wire Line
	5534 1388 5534 1392
Connection ~ 5453 1388
Wire Wire Line
	5453 1388 5453 1409
Wire Wire Line
	5809 1823 6084 1823
Wire Wire Line
	3221 1685 3221 1811
Wire Wire Line
	3221 1811 2508 1811
Connection ~ 2508 1811
Wire Wire Line
	3647 1653 3647 1811
Wire Wire Line
	3647 1811 3221 1811
Connection ~ 3221 1811
Wire Wire Line
	2473 917  3511 917 
Wire Wire Line
	3511 917  3511 540 
Connection ~ 2473 917 
Wire Wire Line
	2473 917  2473 885 
$Comp
L Device:R_Small_US R4
U 1 1 5E491800
P 3663 639
F 0 "R4" H 3731 685 50  0000 L CNN
F 1 "1 KOhm" H 3731 594 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3663 639 50  0001 C CNN
F 3 "~" H 3663 639 50  0001 C CNN
	1    3663 639 
	1    0    0    -1  
$EndComp
Wire Wire Line
	3511 540  3663 540 
Wire Wire Line
	3663 540  3663 539 
Wire Wire Line
	6084 818  5847 818 
Wire Wire Line
	5647 818  5436 818 
Wire Wire Line
	5373 818  5373 740 
Wire Wire Line
	5535 1292 5534 1292
Wire Wire Line
	5436 1292 5436 818 
Connection ~ 5534 1292
Wire Wire Line
	5534 1292 5436 1292
Connection ~ 5436 818 
Wire Wire Line
	5436 818  5373 818 
Wire Wire Line
	3663 540  5373 540 
Connection ~ 3663 540 
Wire Wire Line
	5610 1823 5609 1823
Wire Wire Line
	5029 1823 5029 1379
Connection ~ 5609 1823
Wire Wire Line
	5609 1823 5029 1823
Connection ~ 5029 1379
Wire Wire Line
	5029 1379 5156 1379
Wire Wire Line
	3647 1811 5453 1811
Wire Wire Line
	5453 1811 5453 1705
Connection ~ 3647 1811
Wire Wire Line
	5453 1705 6185 1705
Wire Wire Line
	6185 1705 6185 1374
Connection ~ 5453 1705
Wire Wire Line
	5453 1705 5453 1609
Wire Wire Line
	6304 1342 6304 1341
Wire Wire Line
	6185 1374 6304 1374
Wire Wire Line
	4098 1107 2414 1107
Wire Wire Line
	2414 1107 2414 1066
Wire Wire Line
	2414 1066 2373 1066
Connection ~ 2373 1066
Wire Wire Line
	2373 1066 2373 1100
Wire Wire Line
	4098 1607 3768 1607
Wire Wire Line
	3768 1607 3768 1171
Wire Wire Line
	3768 1171 2625 1171
Connection ~ 2625 1171
Wire Wire Line
	2625 1171 2625 1729
Wire Wire Line
	4098 1107 5748 1107
Wire Wire Line
	5748 1107 5748 1092
Wire Wire Line
	5748 1092 5784 1092
Connection ~ 4098 1107
Wire Wire Line
	4098 1607 5757 1607
Wire Wire Line
	5757 1607 5757 1592
Wire Wire Line
	5757 1592 5784 1592
Connection ~ 4098 1607
$Comp
L Connector:Conn_01x02_Male J3
U 1 1 5E4DB68D
P 6504 1441
F 0 "J3" H 6476 1323 50  0000 R CNN
F 1 "Conn_01x02_Male" H 6476 1414 50  0000 R CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" H 6504 1441 50  0001 C CNN
F 3 "~" H 6504 1441 50  0001 C CNN
	1    6504 1441
	-1   0    0    1   
$EndComp
Wire Wire Line
	6084 1342 6084 1366
Wire Wire Line
	6084 1366 6139 1366
Wire Wire Line
	6139 1366 6139 1335
Wire Wire Line
	6139 833  6084 833 
Wire Wire Line
	6084 833  6084 818 
Connection ~ 6084 1366
Wire Wire Line
	6084 1366 6084 1823
Wire Wire Line
	6139 1335 6262 1335
Wire Wire Line
	6304 1335 6304 1341
Connection ~ 6139 1335
Wire Wire Line
	6139 1335 6139 833 
Connection ~ 6304 1341
$Comp
L Amplifier_Instrumentation:AD620 U4
U 1 1 5EBA4187
P 2800 3334
F 0 "U4" H 3244 3380 50  0000 L CNN
F 1 "AD620" H 3244 3289 50  0000 L CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 2800 3334 50  0001 C CNN
F 3 "https://www.analog.com/media/en/technical-documentation/data-sheets/AD620.pdf" H 2800 3334 50  0001 C CNN
	1    2800 3334
	1    0    0    -1  
$EndComp
$Comp
L oppo:op37G U5
U 1 1 5EBA418D
P 4440 3291
F 0 "U5" H 4465 3722 50  0000 C CNN
F 1 "op37G" H 4465 3631 50  0000 C CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 4440 3291 50  0001 C CNN
F 3 "" H 4440 3291 50  0001 C CNN
	1    4440 3291
	1    0    0    -1  
$EndComp
$Comp
L oppo:op37G U6
U 1 1 5EBA4193
P 6126 3276
F 0 "U6" H 6151 3707 50  0000 C CNN
F 1 "op37G" H 6151 3616 50  0000 C CNN
F 2 "Package_DIP:DIP-8_W7.62mm" H 6126 3276 50  0001 C CNN
F 3 "" H 6126 3276 50  0001 C CNN
	1    6126 3276
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R10
U 1 1 5EBA4199
P 2129 3336
F 0 "R10" H 2197 3382 50  0000 L CNN
F 1 "100 Ohm" H 2197 3291 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 2129 3336 50  0001 C CNN
F 3 "~" H 2129 3336 50  0001 C CNN
	1    2129 3336
	1    0    0    -1  
$EndComp
Wire Wire Line
	2129 3436 2400 3436
Wire Wire Line
	2400 3436 2400 3434
Wire Wire Line
	2129 3236 2401 3236
Wire Wire Line
	2401 3236 2401 3234
Wire Wire Line
	2401 3234 2400 3234
$Comp
L Connector:Conn_01x03_Male J5
U 1 1 5EBA41A4
P 2815 2528
F 0 "J5" V 2877 2672 50  0000 L CNN
F 1 "Conn_01x03_Male" V 2968 2672 50  0000 L CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 2815 2528 50  0001 C CNN
F 3 "~" H 2815 2528 50  0001 C CNN
	1    2815 2528
	0    1    1    0   
$EndComp
Wire Wire Line
	2715 2728 2715 2756
Wire Wire Line
	2715 3034 2700 3034
Wire Wire Line
	2900 3634 2850 3634
Wire Wire Line
	2815 3634 2815 2851
Wire Wire Line
	2700 3634 2700 3663
Wire Wire Line
	2700 3663 2967 3663
Wire Wire Line
	2967 2782 2915 2782
Wire Wire Line
	2915 2782 2915 2728
$Comp
L power:GND #PWR05
U 1 1 5EBA41B2
P 2430 2819
F 0 "#PWR05" H 2430 2569 50  0001 C CNN
F 1 "GND" V 2435 2691 50  0000 R CNN
F 2 "" H 2430 2819 50  0001 C CNN
F 3 "" H 2430 2819 50  0001 C CNN
	1    2430 2819
	0    1    1    0   
$EndComp
$Comp
L power:+9V #PWR06
U 1 1 5EBA41B8
P 2434 2716
F 0 "#PWR06" H 2434 2566 50  0001 C CNN
F 1 "+9V" V 2449 2844 50  0000 L CNN
F 2 "" H 2434 2716 50  0001 C CNN
F 3 "" H 2434 2716 50  0001 C CNN
	1    2434 2716
	0    -1   -1   0   
$EndComp
$Comp
L power:-9V #PWR04
U 1 1 5EBA41BE
P 2426 2918
F 0 "#PWR04" H 2426 2793 50  0001 C CNN
F 1 "-9V" V 2441 3046 50  0000 L CNN
F 2 "" H 2426 2918 50  0001 C CNN
F 3 "" H 2426 2918 50  0001 C CNN
	1    2426 2918
	0    -1   -1   0   
$EndComp
Wire Wire Line
	2434 2716 2434 2756
Wire Wire Line
	2434 2756 2715 2756
Connection ~ 2715 2756
Wire Wire Line
	2715 2756 2715 3000
Wire Wire Line
	2429 2819 2430 2819
Connection ~ 2815 2819
Wire Wire Line
	2815 2819 2815 2728
Connection ~ 2430 2819
Wire Wire Line
	2430 2819 2815 2819
Wire Wire Line
	2426 2918 2967 2918
Wire Wire Line
	2967 2782 2967 2918
Connection ~ 2967 2918
Wire Wire Line
	2967 2918 2967 3105
$Comp
L Connector:Conn_01x03_Male J4
U 1 1 5EBA41D1
P 1434 3312
F 0 "J4" H 1542 3593 50  0000 C CNN
F 1 "Conn_01x03_Male" H 1542 3502 50  0000 C CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical" H 1434 3312 50  0001 C CNN
F 3 "~" H 1434 3312 50  0001 C CNN
	1    1434 3312
	1    0    0    -1  
$EndComp
Wire Wire Line
	1634 3212 1688 3212
Wire Wire Line
	2041 3212 2041 3158
Wire Wire Line
	2041 3158 2400 3158
Wire Wire Line
	2400 3158 2400 3134
Wire Wire Line
	1634 3312 1716 3312
Wire Wire Line
	2040 3312 2040 3551
Wire Wire Line
	2040 3551 2400 3551
Wire Wire Line
	2400 3551 2400 3534
Wire Wire Line
	1634 3412 1741 3412
Wire Wire Line
	1984 3412 1984 3745
Wire Wire Line
	1984 3745 2850 3745
Wire Wire Line
	2850 3745 2850 3634
Connection ~ 2850 3634
Wire Wire Line
	2850 3634 2815 3634
Text GLabel 1424 3555 0    50   Input ~ 0
Electrode1b
Text GLabel 1419 3701 0    50   Input ~ 0
Electrode2b
Text GLabel 1414 3862 0    50   Input ~ 0
reference_Electrode
Wire Wire Line
	1424 3555 1688 3555
Wire Wire Line
	1688 3555 1688 3212
Connection ~ 1688 3212
Wire Wire Line
	1688 3212 2041 3212
Wire Wire Line
	1419 3701 1716 3701
Wire Wire Line
	1716 3701 1716 3312
Connection ~ 1716 3312
Wire Wire Line
	1716 3312 2040 3312
Wire Wire Line
	1414 3862 1741 3862
Wire Wire Line
	1741 3862 1741 3412
Connection ~ 1741 3412
Wire Wire Line
	1741 3412 1984 3412
$Comp
L Device:CP_Small C5
U 1 1 5EBA41F4
P 3381 3333
F 0 "C5" V 3606 3333 50  0000 C CNN
F 1 "0.1 uF" V 3515 3333 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 3381 3333 50  0001 C CNN
F 3 "~" H 3381 3333 50  0001 C CNN
	1    3381 3333
	0    -1   -1   0   
$EndComp
$Comp
L Device:CP_Small C6
U 1 1 5EBA41FA
P 3739 3334
F 0 "C6" V 3964 3334 50  0000 C CNN
F 1 "01 uF" V 3873 3334 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 3739 3334 50  0001 C CNN
F 3 "~" H 3739 3334 50  0001 C CNN
	1    3739 3334
	0    -1   -1   0   
$EndComp
$Comp
L Device:R_Small_US R11
U 1 1 5EBA4200
P 3563 3519
F 0 "R11" H 3631 3565 50  0000 L CNN
F 1 "1.5 MOhm" H 3631 3474 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3563 3519 50  0001 C CNN
F 3 "~" H 3563 3519 50  0001 C CNN
	1    3563 3519
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R12
U 1 1 5EBA4206
P 3989 3487
F 0 "R12" H 4057 3533 50  0000 L CNN
F 1 "1.5 MOhm" H 4057 3442 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 3989 3487 50  0001 C CNN
F 3 "~" H 3989 3487 50  0001 C CNN
	1    3989 3487
	1    0    0    -1  
$EndComp
$Comp
L Device:R_Small_US R14
U 1 1 5EBA420C
P 4200 2889
F 0 "R14" V 3995 2889 50  0000 C CNN
F 1 "3.3 KOhm" V 4086 2889 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 4200 2889 50  0001 C CNN
F 3 "~" H 4200 2889 50  0001 C CNN
	1    4200 2889
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R18
U 1 1 5EBA4212
P 6089 2752
F 0 "R18" V 5884 2752 50  0000 C CNN
F 1 "5 KOhm" V 5975 2752 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 6089 2752 50  0001 C CNN
F 3 "~" H 6089 2752 50  0001 C CNN
	1    6089 2752
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R17
U 1 1 5EBA4218
P 5715 2574
F 0 "R17" H 5783 2620 50  0000 L CNN
F 1 "10 KOhm" H 5783 2529 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5715 2574 50  0001 C CNN
F 3 "~" H 5715 2574 50  0001 C CNN
	1    5715 2574
	1    0    0    -1  
$EndComp
$Comp
L Device:CP_Small C8
U 1 1 5EBA421E
P 6051 3757
F 0 "C8" V 6276 3757 50  0000 C CNN
F 1 "0.1 uF" V 6185 3757 50  0000 C CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 6051 3757 50  0001 C CNN
F 3 "~" H 6051 3757 50  0001 C CNN
	1    6051 3757
	0    -1   -1   0   
$EndComp
$Comp
L Device:CP_Small C7
U 1 1 5EBA4224
P 5795 3443
F 0 "C7" H 5707 3397 50  0000 R CNN
F 1 "0.1 uF" H 5707 3488 50  0000 R CNN
F 2 "Capacitor_THT:CP_Radial_D6.3mm_P2.50mm" H 5795 3443 50  0001 C CNN
F 3 "~" H 5795 3443 50  0001 C CNN
	1    5795 3443
	-1   0    0    1   
$EndComp
$Comp
L Device:R_Small_US R16
U 1 1 5EBA422A
P 5598 3321
F 0 "R16" V 5393 3321 50  0000 C CNN
F 1 "16 KOhm" V 5484 3321 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5598 3321 50  0001 C CNN
F 3 "~" H 5598 3321 50  0001 C CNN
	1    5598 3321
	0    1    1    0   
$EndComp
$Comp
L Device:R_Small_US R15
U 1 1 5EBA4230
P 5123 3313
F 0 "R15" V 4918 3313 50  0000 C CNN
F 1 "16 KOhm" V 5009 3313 50  0000 C CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 5123 3313 50  0001 C CNN
F 3 "~" H 5123 3313 50  0001 C CNN
	1    5123 3313
	0    1    1    0   
$EndComp
Wire Wire Line
	3200 3334 3281 3334
Wire Wire Line
	3281 3334 3281 3333
Wire Wire Line
	3639 3333 3639 3334
Wire Wire Line
	3563 3419 3563 3333
Wire Wire Line
	3481 3333 3563 3333
Connection ~ 3563 3333
Wire Wire Line
	3563 3333 3639 3333
Wire Wire Line
	3839 3334 3989 3334
Wire Wire Line
	4190 3334 4190 3341
Wire Wire Line
	3989 3387 3989 3334
Connection ~ 3989 3334
Wire Wire Line
	3989 3334 4190 3334
Wire Wire Line
	4005 2889 4100 2889
Wire Wire Line
	4300 2889 4782 2889
Wire Wire Line
	4782 2889 4782 3291
Wire Wire Line
	4190 3241 4005 3241
Wire Wire Line
	4005 2673 4005 2889
Connection ~ 4005 2889
Wire Wire Line
	4005 2889 4005 3241
Wire Wire Line
	5023 3291 5023 3313
Wire Wire Line
	4740 3291 4782 3291
Connection ~ 4782 3291
Wire Wire Line
	4782 3291 5023 3291
Wire Wire Line
	5223 3313 5371 3313
Wire Wire Line
	5498 3313 5498 3321
Wire Wire Line
	5698 3321 5795 3321
Wire Wire Line
	5795 3321 5795 3322
Wire Wire Line
	5795 3322 5876 3322
Wire Wire Line
	5876 3322 5876 3326
Connection ~ 5795 3322
Wire Wire Line
	5795 3322 5795 3343
Wire Wire Line
	6151 3757 6426 3757
Wire Wire Line
	3563 3619 3563 3745
Wire Wire Line
	3563 3745 2850 3745
Connection ~ 2850 3745
Wire Wire Line
	3989 3587 3989 3745
Wire Wire Line
	3989 3745 3563 3745
Connection ~ 3563 3745
Wire Wire Line
	2815 2851 3853 2851
Wire Wire Line
	3853 2851 3853 2474
Connection ~ 2815 2851
Wire Wire Line
	2815 2851 2815 2819
$Comp
L Device:R_Small_US R13
U 1 1 5EBA4260
P 4005 2573
F 0 "R13" H 4073 2619 50  0000 L CNN
F 1 "1 KOhm" H 4073 2528 50  0000 L CNN
F 2 "Resistor_THT:R_Axial_DIN0411_L9.9mm_D3.6mm_P12.70mm_Horizontal" H 4005 2573 50  0001 C CNN
F 3 "~" H 4005 2573 50  0001 C CNN
	1    4005 2573
	1    0    0    -1  
$EndComp
Wire Wire Line
	3853 2474 4005 2474
Wire Wire Line
	4005 2474 4005 2473
Wire Wire Line
	6426 2752 6189 2752
Wire Wire Line
	5989 2752 5778 2752
Wire Wire Line
	5715 2752 5715 2674
Wire Wire Line
	5877 3226 5876 3226
Wire Wire Line
	5778 3226 5778 2752
Connection ~ 5876 3226
Wire Wire Line
	5876 3226 5778 3226
Connection ~ 5778 2752
Wire Wire Line
	5778 2752 5715 2752
Wire Wire Line
	4005 2474 5715 2474
Connection ~ 4005 2474
Wire Wire Line
	5952 3757 5951 3757
Wire Wire Line
	5371 3757 5371 3313
Connection ~ 5951 3757
Wire Wire Line
	5951 3757 5371 3757
Connection ~ 5371 3313
Wire Wire Line
	5371 3313 5498 3313
Wire Wire Line
	3989 3745 5795 3745
Wire Wire Line
	5795 3745 5795 3639
Connection ~ 3989 3745
Wire Wire Line
	5795 3639 6527 3639
Wire Wire Line
	6527 3639 6527 3308
Connection ~ 5795 3639
Wire Wire Line
	5795 3639 5795 3543
Wire Wire Line
	6646 3276 6646 3275
Wire Wire Line
	6646 3308 6646 3375
Wire Wire Line
	6527 3308 6646 3308
Wire Wire Line
	4440 3041 2756 3041
Wire Wire Line
	2756 3041 2756 3000
Wire Wire Line
	2756 3000 2715 3000
Connection ~ 2715 3000
Wire Wire Line
	2715 3000 2715 3034
Wire Wire Line
	4440 3541 4110 3541
Wire Wire Line
	4110 3541 4110 3105
Wire Wire Line
	4110 3105 2967 3105
Connection ~ 2967 3105
Wire Wire Line
	2967 3105 2967 3663
Wire Wire Line
	4440 3041 6090 3041
Wire Wire Line
	6090 3041 6090 3026
Wire Wire Line
	6090 3026 6126 3026
Connection ~ 4440 3041
Wire Wire Line
	4440 3541 6099 3541
Wire Wire Line
	6099 3541 6099 3526
Wire Wire Line
	6099 3526 6126 3526
Connection ~ 4440 3541
$Comp
L Connector:Conn_01x02_Male J6
U 1 1 5EBA4295
P 6846 3375
F 0 "J6" H 6818 3257 50  0000 R CNN
F 1 "Conn_01x02_Male" H 6818 3348 50  0000 R CNN
F 2 "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical" H 6846 3375 50  0001 C CNN
F 3 "~" H 6846 3375 50  0001 C CNN
	1    6846 3375
	-1   0    0    1   
$EndComp
Wire Wire Line
	6426 3276 6426 3300
Wire Wire Line
	6426 3300 6481 3300
Wire Wire Line
	6481 3300 6481 3269
Wire Wire Line
	6481 2767 6426 2767
Wire Wire Line
	6426 2767 6426 2752
Connection ~ 6426 3300
Wire Wire Line
	6426 3300 6426 3757
Wire Wire Line
	6481 3269 6646 3269
Wire Wire Line
	6646 3269 6646 3275
Connection ~ 6481 3269
Wire Wire Line
	6481 3269 6481 2767
Connection ~ 6646 3275
Wire Notes Line
	1577 1141 1577 1945
Wire Notes Line
	1577 1945 2845 1945
Wire Notes Line
	2845 1945 2845 1019
Wire Notes Line
	2845 1019 1577 1019
Wire Notes Line
	1577 1019 1577 1138
Wire Notes Line
	2902 1294 2902 1925
Wire Notes Line
	2902 1925 3713 1925
Wire Notes Line
	3713 1925 3713 1294
Wire Notes Line
	3713 1294 2901 1294
Wire Notes Line
	3459 509  3459 1240
Wire Notes Line
	3459 1240 3739 1240
Wire Notes Line
	3739 1240 3739 1923
Wire Notes Line
	3739 1923 4517 1923
Wire Notes Line
	4517 1923 4517 840 
Wire Notes Line
	4517 840  4058 840 
Wire Notes Line
	4058 840  4058 505 
Wire Notes Line
	4058 505  3459 505 
Wire Notes Line
	4562 1058 4562 1925
Wire Notes Line
	4562 1925 5940 1925
Wire Notes Line
	5940 1925 5940 1671
Wire Notes Line
	5940 1671 5531 1671
Wire Notes Line
	5531 1671 5531 1316
Wire Notes Line
	5531 1316 5418 1316
Wire Notes Line
	5418 1316 5418 1058
Wire Notes Line
	5418 1058 4562 1058
Wire Notes Line
	5297 499  5297 1027
Wire Notes Line
	5297 1027 5449 1027
Wire Notes Line
	5449 1027 5449 1284
Wire Notes Line
	5449 1284 5559 1284
Wire Notes Line
	5559 1284 5559 1642
Wire Notes Line
	5559 1642 5968 1642
Wire Notes Line
	5968 1642 5968 1922
Wire Notes Line
	5968 1922 6188 1922
Wire Notes Line
	6188 1922 6188 491 
Wire Notes Line
	6188 491  5297 491 
Text Label 1608 1934 0    50   ~ 0
Instru._Amp.
Text Label 2917 1908 0    50   ~ 0
HighPassFilter
Text Label 3753 1909 0    50   ~ 0
Amplifier
Text Label 4577 1906 0    50   ~ 0
LowPassFilter
Wire Notes Line
	2116 4004 3169 4004
Wire Notes Line
	3169 4004 3169 2964
Wire Notes Line
	3169 2964 2116 2964
Wire Notes Line
	2116 2964 2116 4004
Wire Notes Line
	3310 2817 3310 3999
Wire Notes Line
	3310 3999 4905 3999
Wire Notes Line
	4905 3999 4905 2379
Wire Notes Line
	4905 2379 3311 2379
Wire Notes Line
	3311 2379 3311 2824
Wire Notes Line
	4963 2386 4963 4002
Wire Notes Line
	4963 4002 6538 4002
Wire Notes Line
	6538 4002 6538 2384
Wire Notes Line
	6538 2384 4967 2384
Wire Wire Line
	1301 2048 1400 2048
Wire Wire Line
	1400 2048 1400 1929
Wire Wire Line
	1400 1929 1399 1929
Wire Wire Line
	1399 1478 1399 1929
Text GLabel 6340 1209 2    50   Output ~ 0
ToADC
Text GLabel 6318 1507 2    50   Output ~ 0
GND
Wire Wire Line
	6318 1507 6304 1507
Wire Wire Line
	6304 1374 6304 1441
Connection ~ 6304 1441
Wire Wire Line
	6304 1441 6304 1507
Wire Wire Line
	6340 1208 6262 1208
Wire Wire Line
	6262 1208 6262 1335
Connection ~ 6262 1335
Wire Wire Line
	6262 1335 6304 1335
$EndSCHEMATC