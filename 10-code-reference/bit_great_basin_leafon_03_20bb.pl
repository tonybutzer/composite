#!/usr/bin/perl

$dosrootdir = "/caldera/projects/usgs/eros/users/bbunde/";
$batchDir = "/caldera/projects/usgs/eros/users/bbunde/batch/";
$jobDir = "/caldera/projects/usgs/eros/users/bbunde/jobs/";
print "$dosrootdir\n";

#Arrays
#@blocks = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50);
#@blocks = (48,43,44,47,24,25,19,17,1,2,14,46,45,50,49,39,40,42,34,35,36,37,38,29,30,31,32,33,26,27,28,20,21,22,23,15,16,18,11,12,13,8,9,10,4,5,6,7,3);
#@blocks = (41,1,2,14,17,19,24,25,47,48);
@blocks = ("GreatBasin_Region");
#@blocks = (1,2,5,6,8,9,11,12,13,15,16,17,19,20,21,24,25,26,29,30,31,34,35,36,39,40,43,44,48,49,50);
@years = (2003,2004,2005,2006,2007,2008,2009,2010,2011,2013,2014,2015,2016,2017,2018,2019,2020);
#@years = ("2008");
$composite_type = ("leafon");
$sensors = ("l5,l8");
foreach $blk (@blocks) {
	$count = 0;

	foreach $yr (@years) {
		@output = ();
		@date = ();
		@sname=();
		#$count = 0; 
                $arraynum = 0;
		
	        #leafon Parameters	
		$output_leafon = $dosrootdir."output/".$blk."_leafonn_".$yr."_0315_0615.img";
	        $length_leafon = length($output_leafon);
		#print "Length is: $length_leafon\n";
		$leafon_name = substr ($output_leafon,$length_leafon -16, -4);
		print $leafon_name;
                $file_outputname2 = $blk."_".$leafon_name;
		print "Final Name is: $file_outputname2\n";
                push (@sname, $file_outputname2);

		push (@output,$output_leafon);
		#print "Output leafon is: $output_leafon\n";
		
	
		if ($yr == 2003) {
			$leafon_date = "2003-03-15/2003-06-15";
			
        		
		}elsif ($yr == 2004) {
			$leafon_date = "2004-03-15/2004-06-15";
			
     
		}elsif ($yr == 2005) {
			$leafon_date = "2005-03-15/2005-06-15";
				
		}elsif ($yr == 2006) {
			$leafon_date = "2006-03-15/2006-06-15";
			
     
		}elsif ($yr == 2007) {
			$leafon_date = "2007-03-15/2007-06-15";
		
		}elsif ($yr == 2008) {
			$leafon_date = "2008-03-15/2008-06-15";
			
     
		}elsif ($yr == 2009) {
			$leafon_date = "2009-03-15/2009-06-15";
				
		}elsif ($yr == 2010) {
			$leafon_date = "2010-03-15/2010-06-15";
			
     
		}elsif ($yr == 2011) {
			$leafon_date = "2011-03-15/2011-06-15";
			
		}elsif ($yr == 2013) {
			$leafon_date = "2013-03-15/2013-06-15";
			
     
		}elsif ($yr == 2014) {
			$leafon_date = "2014-03-15/2014-06-15";
				
		}elsif ($yr == 2015) {
			$leafon_date = "2015-03-15/2015-06-15";
			
     
		}elsif ($yr == 2016) {
			$leafon_date = "2015-03-15/2015-06-15,2016-03-15/2016-06-15,2017-03-15/2017-06-15";
		
		}elsif ($yr == 2017) {
			$leafon_date = "2017-03-15/2017-06-15";
			
     
		}elsif ($yr == 2018) {
			$leafon_date = "2018-03-15/2018-06-15";
				
		}elsif ($yr == 2019) {
			$leafon_date = "2019-03-15/2019-06-15";
			
     
		}elsif ($yr == 2020) {
			$leafon_date = "2020-03-15/2020-06-15";
			
			
			
		}
		
		
		#print "leafon Dates: $leafon_date\n";
		push (@date,$leafon_date);


                for (0 .. 0) {
                         slurm();

               
			 #print "arraynum is: $arraynum\n";
			 print "Count is: $count\n"; 

                if ($count == 30) {
			print "Kicked off all jobs.  Job counter is $count\n"; 
		checkJobs();
		while(1){
			if ( $x == 0) {
                                print "\nGenerating ARD Composites for $blk and Year $yr.  There are currently $x compositing\n";
                                last;
                        }else {
                                print "Can't kicked any more composites all $x in use\n";
                                sleep (50);
                                checkJobs();
		
                        }
		}
               
            }
	    $arraynum++;
	    $count++;
 
           }	
	}	  
}

#clear array
@blocks = ();
@years = ();
@date =();
@output = ();
@composite_type = ();


sub slurm {

	$slmfile = $batchDir."$sname[$arraynum]\.slm";
	print "Slurm file is: $slmfile and  $yr\n";
	open(OUTFILE, ">$slmfile");
        print OUTFILE "#!/bin/bash\n";	
	print OUTFILE "#SBATCH -A eros\n";
	print OUTFILE "#SBATCH -p workq\n";
	print OUTFILE "#SBATCH --job-name=gb_leafon_03_20\n";	
	print OUTFILE "#SBATCH --output=".$jobDir."gb_leafon_03_20-%j.out\n";
	print OUTFILE "#SBATCH --error=".$jobDir."gb_leafon_03_20-%j.err\n";
	print OUTIFLE "#SBATCH --nodes=1\n";
	print OUTFILE "#SBATCH --ntasks=1\n";
	print OUTFILE "#SBATCH --cpus-per-task=40\n";
	print OUTFILE "#SBATCH --hint=nomultithread\n";
	print OUTFILE "#SBATCH --time=48:00:00\n";
	print OUTFILE "#SBATCH --mail-user=bbunde\@contractor.usgs.gov\n";
	print OUTFILE "#SBATCH --mail-type=ALL\n";


	print OUTFILE "#!/home/pdanielson/miniconda3-bash.sh\n";
	print OUTFILE "source /home/pdanielson/miniconda3-bash.sh\n";
	print OUTFILE "conda activate py38\n";
	print OUTFILE "python /caldera/projects/usgs/eros/users/bbunde/scripts/bit_v4/bit_v4_1600.py /caldera/projects/usgs/eros/users/bbunde/shapefiles/".$blk."\.shp $output[$arraynum] $date[$arraynum] -sensors $sensors -cpu \$SLURM_CPUS_PER_TASK\n";
	close(OUTFILE);
	system("sbatch $slmfile");
	return;

}

sub checkJobs {
	$checkfile = $batchDir."process\.out";
	system ("squeue -u bbunde> $checkfile");
	open(INFILE, "<$checkfile") or die "+++ err unable to open $checkfile +++\n";
	while (defined($theLine2 = <INFILE>)) {
		chomp($theLine2);
		push(@input2,$theLine2);
	}#end while
	close(INFILE);
	
	$x = 0;
	foreach $line2 (@input2) {
		chomp ($line2);
		if ($line2 =~ m/bbunde/) {
			$x = $x + 1;
		}
	}
	@input2 =();
return $x

}	
